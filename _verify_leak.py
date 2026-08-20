from requests.models import Response, PreparedRequest

r = PreparedRequest()
r.prepare(
    method="POST",
    url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
    params={"key": "AIzaSyFAKEKEY1234567890"},
)
print("prepared url:", r.url)
resp = Response()
resp.status_code = 400
resp.reason = "Bad Request"
resp.url = r.url
try:
    resp.raise_for_status()
except Exception as e:
    print("exc str:", str(e))
    print("formatted like log.warning:", "Gemini attempt %d/%d failed (%s)" % (1, 6, e))
