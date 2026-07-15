"""taiji.life — 态极生命系统

包含：
- life_scheduler: 生命调度器（心跳循环）
- feed_engine:    喂养引擎（吃饭）
- sleep_engine:   睡眠引擎（睡觉）
- play_engine:    玩耍引擎（娱乐）
- evolution_engine: 进化引擎

注意：身体模块（body/limbs/metabolism/senses）已迁移至 taiji.body 包。
"""
from taiji.life.life_scheduler import *  # noqa: F401,F403
from taiji.life.feed_engine import *  # noqa: F401,F403
from taiji.life.sleep_engine import *  # noqa: F401,F403
from taiji.life.play_engine import *  # noqa: F401,F403
from taiji.life.evolution_engine import *  # noqa: F401,F403

# 身体模块（已迁移至 taiji.body，按需延迟导入）
from taiji.body.core import BodyCore  # noqa: F401
