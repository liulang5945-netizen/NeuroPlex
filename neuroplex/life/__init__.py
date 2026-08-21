"""neuroplex.life — Legacy NeuroPlex 生命系统

包含：
- life_scheduler: 生命调度器（心跳循环）
- feed_engine:    喂养引擎（吃饭）
- sleep_engine:   睡眠引擎（睡觉）
- play_engine:    玩耍引擎（娱乐）
- evolution_engine: 进化引擎

注意：身体模块（body/limbs/metabolism/senses）位于 neuroplex.body 包。
"""
from neuroplex.life.life_scheduler import *  # noqa: F401,F403
from neuroplex.life.feed_engine import *  # noqa: F401,F403
from neuroplex.life.sleep_engine import *  # noqa: F401,F403
from neuroplex.life.play_engine import *  # noqa: F401,F403
from neuroplex.life.evolution_engine import *  # noqa: F401,F403

# 身体模块（neuroplex.body，按需延迟导入）
from neuroplex.body.core import BodyCore  # noqa: F401
