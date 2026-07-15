"""
Taiji 安全模块
================
提供 JWT 认证、敏感数据加密、操作审计等功能。
纯 Python 实现，不引入额外外部依赖。

主要组件:
- JWTManager: 轻量级 JWT 管理（hmac + hashlib）
- SecureStorage: 敏感数据加密存储
- AuditLogger: 安全操作审计日志
"""
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import time
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from taiji.services.settings_service import load_settings, update_settings

logger = logging.getLogger("Security")

# ======================== 路径工具 ========================

def _security_dir() -> str:
    """安全数据存储目录"""
    from taiji.core.config import get_external_path
    path = get_external_path("security")
    os.makedirs(path, exist_ok=True)
    return path


# ======================== JWT 认证 ========================

class JWTManager:
    """轻量级 JWT 管理器（纯 Python hmac 实现，无外部依赖）

    支持:
    - 创建 Token（HS256 签名）
    - 验证 Token（签名 + 过期时间检查）
    - 刷新即将过期的 Token
    """

    def __init__(self, secret_key: str = None, token_expire_hours: int = 24):
        self.secret_key = secret_key or self._load_or_generate_secret()
        self.token_expire_hours = token_expire_hours

    def _secret_path(self) -> str:
        return os.path.join(_security_dir(), ".jwt_secret")

    def _load_or_generate_secret(self) -> str:
        """从安全存储加载或生成 JWT 密钥"""
        key_path = self._secret_path()
        if os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    key = f.read().strip()
                if len(key) >= 32:
                    return key
            except Exception:
                pass
        # 生成新密钥
        secret = secrets.token_hex(32)
        try:
            with open(key_path, "w") as f:
                f.write(secret)
            # 限制文件权限
            try:
                os.chmod(key_path, 0o600)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"保存 JWT 密钥失败: {e}")
        return secret

    def _b64_encode(self, data: bytes) -> str:
        """URL-safe Base64 编码（去 padding）"""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _b64_decode(self, s: str) -> bytes:
        """URL-safe Base64 解码（补 padding）"""
        s += "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(s)

    def _sign(self, message: str) -> str:
        """HS256 签名"""
        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).digest()
        return self._b64_encode(sig)

    def create_token(self, username: str, extra_claims: dict = None) -> str:
        """创建 JWT Token"""
        # Header
        header = self._b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())

        # Payload
        now = int(time.time())
        payload_data = {
            "sub": username,
            "iat": now,
            "exp": now + self.token_expire_hours * 3600,
        }
        if extra_claims:
            payload_data.update(extra_claims)
        payload = self._b64_encode(json.dumps(payload_data).encode())

        # Signature
        message = f"{header}.{payload}"
        signature = self._sign(message)

        return f"{message}.{signature}"

    def verify_token(self, token: str) -> Optional[dict]:
        """验证 JWT Token，返回 payload 或 None"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header, payload, signature = parts

            # 验证签名
            expected_sig = self._sign(f"{header}.{payload}")
            if not hmac.compare_digest(signature, expected_sig):
                logger.warning("JWT 签名验证失败")
                return None

            # 解码 payload
            payload_data = json.loads(self._b64_decode(payload))

            # 检查过期时间
            if "exp" in payload_data:
                if time.time() > payload_data["exp"]:
                    logger.info("JWT Token 已过期")
                    return None

            return payload_data
        except Exception as e:
            logger.warning(f"JWT 验证失败: {e}")
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """刷新即将过期的 Token（过期前 2 小时内可刷新）"""
        payload = self.verify_token(token)
        if not payload:
            return None

        # 只在过期前 2 小时内允许刷新
        exp = payload.get("exp", 0)
        if time.time() > exp - 2 * 3600:
            return self.create_token(
                payload.get("sub", "unknown"),
                {k: v for k, v in payload.items() if k not in ("sub", "iat", "exp")}
            )
        return None  # Token 还很新鲜，不需要刷新


# ======================== 敏感数据加密 ========================

class SecureStorage:
    """敏感数据加密存储

    使用 XOR + HMAC 的轻量级加密方案（无需 cryptography 库）。
    适用于保护 API Key 等短文本。
    """

    def __init__(self):
        self._master_key = self._derive_master_key()

    def _machine_fingerprint(self) -> str:
        """生成机器指纹（CPU + 用户名 + 主机名）"""
        import platform
        parts = [
            platform.node(),
            platform.machine(),
            os.environ.get("USERNAME", os.environ.get("USER", "")),
            platform.processor() or "unknown",
        ]
        return "|".join(parts)

    def _derive_master_key(self) -> bytes:
        """从机器指纹派生主密钥"""
        fingerprint = self._machine_fingerprint()
        # 多轮哈希增强安全性
        key = fingerprint.encode("utf-8")
        for _ in range(10000):
            key = hashlib.sha256(key + b"taiji-salt-2024").digest()
        return key  # 32 bytes

    def encrypt(self, plaintext: str) -> str:
        """加密字符串，返回 Base64 编码的密文"""
        if not plaintext:
            return ""

        data = plaintext.encode("utf-8")
        # 生成随机 IV
        iv = secrets.token_bytes(16)

        # XOR 加密（使用 master_key 派生密钥流）
        key_stream = self._generate_keystream(iv, len(data))
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))

        # 计算 HMAC 用于完整性验证
        mac = hmac.new(self._master_key, iv + encrypted, hashlib.sha256).digest()[:16]

        # 组合: IV(16) + MAC(16) + Encrypted
        result = iv + mac + encrypted
        return base64.b64encode(result).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """解密 Base64 编码的密文"""
        if not ciphertext:
            return ""

        try:
            raw = base64.b64decode(ciphertext)
            if len(raw) < 32:
                return ""

            iv = raw[:16]
            mac = raw[16:32]
            encrypted = raw[32:]

            # 验证 HMAC
            expected_mac = hmac.new(self._master_key, iv + encrypted, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(mac, expected_mac):
                logger.warning("解密失败: HMAC 验证不通过")
                return ""

            # XOR 解密
            key_stream = self._generate_keystream(iv, len(encrypted))
            decrypted = bytes(a ^ b for a, b in zip(encrypted, key_stream))
            return decrypted.decode("utf-8")
        except Exception as e:
            logger.warning(f"解密失败: {e}")
            return ""

    def _generate_keystream(self, iv: bytes, length: int) -> bytes:
        """使用 HMAC 生成密钥流"""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hmac.new(
                self._master_key,
                iv + struct.pack(">I", counter),
                hashlib.sha256
            ).digest()
            stream += block
            counter += 1
        return stream[:length]


# ======================== 操作审计 ========================

class AuditLogger:
    """安全操作审计日志"""

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(_security_dir(), "audit_logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        """当天的日志文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{today}.jsonl")

    def log_event(self, event_type: str, detail: dict = None,
                  user: str = "local", ip: str = "127.0.0.1"):
        """记录安全事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "user": user,
            "ip": ip,
            "detail": detail or {},
        }
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"审计日志写入失败: {e}")

    def get_recent_events(self, limit: int = 100, days: int = 7) -> List[dict]:
        """获取最近的审计事件"""
        events = []
        try:
            from datetime import timedelta
            today = datetime.now()
            for d in range(days):
                day = today - timedelta(days=d)
                path = os.path.join(self.log_dir, f"audit_{day.strftime('%Y-%m-%d')}.jsonl")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    events.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
            # 按时间降序排列
            events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.warning(f"读取审计日志失败: {e}")
        return events[:limit]


# ======================== 认证管理器 ========================

class AuthManager:
    """认证管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jwt = JWTManager()
            cls._instance._storage = SecureStorage()
            cls._instance._audit = AuditLogger()
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self):
        """从 app_settings.json 加载认证配置"""
        self.enabled = False
        self.username = "admin"
        self.password_hash = ""
        try:
            data = load_settings()
            self.enabled = data.get("auth_enabled", False)
            self.username = data.get("auth_username", "admin")
            self.password_hash = data.get("auth_password_hash", "")
        except Exception:
            pass

    def _save_settings(self):
        """保存认证配置"""
        try:
            update_settings({
                "auth_enabled": self.enabled,
                "auth_username": self.username,
                "auth_password_hash": self.password_hash,
            })
        except Exception as e:
            logger.warning(f"保存认证配置失败: {e}")

    @staticmethod
    def _hash_password(password: str, salt: str = None) -> str:
        """密码哈希（SHA-256 + 随机 salt）"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return f"{salt}${hashed.hex()}"

    def set_password(self, password: str):
        """设置密码（使用随机 salt）"""
        self.password_hash = self._hash_password(password)
        self._save_settings()
        self._audit.log_event("password_changed", {"user": self.username})

    def verify_password(self, password: str) -> bool:
        """验证密码（支持新格式 salt$hash 和旧格式纯哈希）"""
        if not self.password_hash:
            return True  # 未设置密码时默认通过
        if "$" in self.password_hash:
            # 新格式: salt$hash
            salt, stored_hash = self.password_hash.split("$", 1)
            computed = self._hash_password(password, salt=salt)
            _, computed_hash = computed.split("$", 1)
            return hmac.compare_digest(computed_hash, stored_hash)
        else:
            # 旧格式兼容: 纯 SHA-256 哈希
            legacy_salt = "taiji-pwd-salt-2024"
            legacy_hash = hashlib.sha256(f"{legacy_salt}{password}".encode()).hexdigest()
            return hmac.compare_digest(legacy_hash, self.password_hash)

    def login(self, username: str, password: str) -> Optional[str]:
        """用户登录，返回 JWT Token 或 None。

        安全策略：
        - 认证未启用时：仅当已设置密码时才允许无认证访问（防止默认开放）
        - 认证启用时：必须提供正确的用户名和密码
        """
        if not self.enabled:
            # 如果已设置密码，未启用认证可能是部署失误，拒绝登录并提示
            if self.password_hash:
                self._audit.log_event("login_failed", {"user": username, "reason": "auth_disabled_but_password_set"})
                return None
            # 密码也未设置 = 全新部署首次使用，允许无密码访问
            self._audit.log_event("login_noauth_first_run", {"user": username})
            return self._jwt.create_token(username)

        if not self.password_hash:
            # 认证已启用但未设置密码 → 配置错误
            self._audit.log_event("login_failed", {"user": username, "reason": "password_not_set"})
            return None

        if username != self.username:
            self._audit.log_event("login_failed", {"user": username, "reason": "invalid_user"})
            return None

        if not self.verify_password(password):
            self._audit.log_event("login_failed", {"user": username, "reason": "invalid_password"})
            return None

        token = self._jwt.create_token(username)
        self._audit.log_event("login_success", {"user": username})
        return token

    def verify_token(self, token: str) -> Optional[dict]:
        """验证 Token"""
        return self._jwt.verify_token(token)

    def enable_auth(self, username: str, password: str):
        """启用认证"""
        self.enabled = True
        self.username = username
        self.set_password(password)
        self._save_settings()
        self._audit.log_event("auth_enabled", {"user": username})

    def disable_auth(self):
        """禁用认证"""
        self.enabled = False
        self._save_settings()
        self._audit.log_event("auth_disabled", {"user": self.username})

    def get_status(self) -> dict:
        """获取认证状态"""
        return {
            "enabled": self.enabled,
            "username": self.username,
            "has_password": bool(self.password_hash),
        }

    @property
    def jwt(self) -> JWTManager:
        return self._jwt

    @property
    def audit(self) -> AuditLogger:
        return self._audit

    @property
    def storage(self) -> SecureStorage:
        return self._storage