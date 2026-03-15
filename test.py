from mcp_server import scrape_web

url = "https://github.com/alexeygrigorev/minsearch"

content = scrape_web(url)

print("Characters returned:", len(content))
