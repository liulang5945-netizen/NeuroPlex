import urllib.request, json, time, sys

BASE = 'http://127.0.0.1:8000'

def api(path, data=None, method=None):
    try:
        if method is None:
            method = 'POST' if data else 'GET'
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(f'{BASE}{path}', data=body,
            headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {'error': str(e)}
    except Exception as e:
        return 0, {'error': str(e)}

# 1. GET /api/chat/stream
s, d = api('/api/chat/stream', method='GET')
print('--- /api/chat/stream GET ---')
print(s, json.dumps(d, ensure_ascii=False))
# 2. POST /api/chat/sessions
s2, d2 = api('/api/chat/sessions', {'id': 99902, 'name': 'API测试会话'})
print('--- /api/chat/sessions POST ---')
print(s2, json.dumps(d2, ensure_ascii=False))
# 3. GET /api/health
s3, d3 = api('/api/health')
print('--- /api/health GET ---')
print(s3, json.dumps(d3, ensure_ascii=False))