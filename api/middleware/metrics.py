"""Prometheus指标中间件"""
from prometheus_client import Counter, Histogram, Gauge
import time

# 定义指标
requests_total = Counter(
    'taiji_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'taiji_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

errors_total = Counter(
    'taiji_errors_total',
    'Total errors',
    ['error_type']
)

active_connections = Gauge(
    'taiji_active_connections',
    'Active connections'
)

async def metrics_middleware(request, call_next):
    """记录请求指标"""
    active_connections.inc()
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        
        requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    except Exception as e:
        errors_total.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        active_connections.dec()