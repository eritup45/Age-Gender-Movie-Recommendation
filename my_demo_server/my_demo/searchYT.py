import urllib
import re

def get_youtube_search_url(query):
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}&sp=EgIQAQ%253D%253D"

#html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword)
#video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
#webbrowser.open_new("https://www.youtube.com/watch?v=" + video_ids[0])

#https://codefather.tech/blog/youtube-search-python/
