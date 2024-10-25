# back_end_service/crawlings/settings.py
ROTATING_PROXY_LIST = [
    'proxy1:port',
    'proxy2:port',
    # Add more proxies
]

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
    'scrapy_rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
    'scrapy_rotating_proxies.middlewares.BanDetectionMiddleware': 620,
}
