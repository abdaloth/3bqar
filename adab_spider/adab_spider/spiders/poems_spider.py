import scrapy
import logging
from adab_spider.items import Poem

class PoemSpider(scrapy.Spider):
    name = "adab_poems"
    # poet index page
    start_urls = ['http://adab.com/modules.php?name=Sh3er&file=search&doWhat=fast&query=&searchwhat=sha3r&r=asc&start=0']
    allowed_domains = ['adab.com']


    def parse(self, response):
        for a in response.css('center center td:nth-child(1) a::attr(href)').extract():
            yield scrapy.Request('http://adab.com/{}'.format(a),  self.parse_poet)
            logging.debug('yielding next poetfrom href: {}'.format(a))
        next_page = response.css('.tiny:last-child::attr(href)').extract_first()
        logging.debug('(parse) next_page value: {}'.format(next_page))
        if(next_page):
            yield scrapy.Request('http://adab.com/{}'.format(next_page), callback=self.parse)


    def parse_poet(self, response):
        for a in response.css('td td td center tr+ tr a::attr(href)').extract():
            yield scrapy.Request('http://adab.com/{}'.format(a), self.parse_poem)
            logging.debug('yielding next poem from href: {}'.format(a))
        next_page = response.css('.tiny:last-child::attr(href)').extract_first()
        logging.debug('(parse_poet) next_page value: {}'.format(next_page))
        if(next_page):
            yield scrapy.Request('http://adab.com/{}'.format(next_page), callback=self.parse_poet)


    def parse_poem(self, response):
        # NOTE: returned the full html element on some instances
        title = response.css('.poemt::text').extract_first() 
        section, author = response.css('table+ table a+ a::text').extract()

        poem_text = (' ').join(response.css('.poem::text').extract())

        yield Poem(author=author,title=title,section=section,poem_text=poem_text)
