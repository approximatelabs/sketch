wget --recursive --no-parent https://analytics.wikimedia.org/published/datasets/one-off/wikidata/sparql_query_logs/

wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.ttl.gz

# I just learned something, I think anyone with a "wikipedia" hosted thing (??) could do this
# there was a url `your.org` thing...

# or at least, `dumps.wikimedia.your.org` might be a thing, that serves up data that the code can parse
# and turn into a database (eg. load that into sqlite, load it into sparql query, it's accessable) 