import urllib.request

html = urllib.request.urlopen('http://127.0.0.1:5000/').read().decode('utf-8', 'ignore')
print('Afghanistan' in html)
print('Asses' in html)
print('Livestock Food Prediction' in html)
start = html.find('<select id="Area"')
print(html[start:start + 1200])
