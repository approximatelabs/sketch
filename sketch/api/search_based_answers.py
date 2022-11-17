import os

import aiohttp
from bs4 import BeautifulSoup

from sketch.examples.prompt_machine import *


# https://www.googleapis.com/customsearch/v1?[parameters]
# LIMITS: 10k/day
# also equivalent to ~$50 a day
# that's 1 every 8.6s -> or, if burst (2 hours of conversation) -> one every ~0.7 seconds.
# hmm, that's not so bad.
@prompt
async def get_google_response(query):
    google_search_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", None)
    google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", None)
    assert (
        google_search_api_key is not None
    ), "Must have a GOOGLE_SEARCH_API_KEY (and possibly GOOGLE_SEARCH_ENGINE_ID)"
    assert google_search_engine_id is not None, "Must have a GOOGLE_SEARCH_ENGINE_ID"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": google_search_api_key,
                "cx": "52cad330ac26249b7",
                "q": query,
            },
            timeout=5,
        ) as resp:
            response = await resp.json()
    return response


deeper_website_choice = asyncGPT3Prompt(
    "deeper_website_choice",
    """
For the question [{{ question }}], the search results are
{{ search_results }}
In order to answer the question, which three page indices (0-9 from above) should be further investigated? (eg. [2, 7, 9])
[""",
    stop="]",
)


def extract_text(html):
    soup = BeautifulSoup(html, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


@prompt
async def get_more_info_from_webiste(question, url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            html = await resp.text()
    return extract_text(html)[:4000]


get_question_answer_from_website = asyncGPT3Prompt(
    "get_question_answer_from_website",
    """
For the question [{{ question }}], the website [{{ url }}] was chosen.
The possibly truncated text from the website is
```
{{ text }}
```
What is the answer to the question, with defense and quotes from the text? (The goal is to be truthful and reliable)
""",
)


@prompt
async def get_answer_to_question_based_on_website(question, url):
    text = await get_more_info_from_webiste(question, url)
    return await get_question_answer_from_website(question=question, url=url, text=text)


google_based_answer = asyncGPT3Prompt(
    "google_based_answer",
    """
QUESTION AND ANSWER -- Google Search Results Based
 (The goal is to be truthful and reliable, so if the answer isn't possible it is best to explain the difficulty with answering wrather than trying to answer.
  If there is a clear answer (multiple pages agree, and summaries of some pages also agree) then explain the answer and the evidence for it.)
QUESTION: {{ question }}
For the question [{{ question }}], the search results are
====
{{ search_results }}
====
Additionally, 3 pages were explored, and their summaries and answers are:
==
{{ website_answers }}
==
What is the answer to the question ({{question}}), with defense and quotes from the results above?
""",
)


@prompt
async def google_answer_to_question(question):
    raw_results = await get_google_response(question)
    search_results = "\n".join(
        [
            f"{i}: {x['title']} ({x['snippet']})"
            for i, x in enumerate(raw_results["items"])
        ]
    )
    next_choices = await deeper_website_choice(
        question=question, search_results=search_results
    )
    try:
        next_choices = [int(x) for x in next_choices.split(",")]
    except:
        next_choices = [0, 1, 2]
    urls = [raw_results["items"][x]["link"] for x in next_choices]
    # gather seems to fail when inspecting??
    # answers = await asyncio.gather(*[get_answer_to_question_based_on_website(question, url) for url in urls])
    futures = [
        get_answer_to_question_based_on_website(question=question, url=url)
        for url in urls
    ]
    answers = []
    for future in futures:
        try:
            answers.append(await future)
        except Exception as e:
            answers.append("ERROR: " + str(e))
    return await google_based_answer(
        question=question,
        search_results=search_results,
        website_answers="\n".join(answers),
    )
