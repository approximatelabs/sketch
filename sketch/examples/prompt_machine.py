import asyncio
import inspect
import json
import os
import sqlite3
import time
import uuid
from hashlib import md5
from re import S

import requests
import websockets
from dotenv import load_dotenv
from jinja2 import Environment, meta

# This is a very clear work in progress, just trying to move some code into a place where it can be cleaned later, but for now re-used.

load_dotenv()


PM_SETTINGS = {
    "BOT_TOKEN": "5a3556bb2de44a73ab2e5643cb633a6c", 
    "THREAD_ID": "default",
    "DB_PATH": "../datasets/nba_sql.db",
    "LOCAL_HISTORY_DB": "sqlite+aiosqlite:///promptHistory.db",
    "VERBOSE": True,
    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
}
PM_SETTINGS['uri'] = f"wss://www.approx.dev/ws/chat?thread_id={PM_SETTINGS['THREAD_ID']}"

env = Environment()


def get_gpt3_response(prompt, temperature=0, stop=None):
    if not PM_SETTINGS['openai_api_key']:
        raise Exception("No OpenAI API key found")
    # print the prompt if verbose mode
    if PM_SETTINGS['VERBOSE']:
        print(prompt)
    headers = {
        "Authorization": f"Bearer {PM_SETTINGS['openai_api_key']}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": temperature,
        "model": "text-davinci-002",
    }
    if stop:
        data["stop"] = stop
    response = requests.post(
        "https://api.openai.com/v1/completions", headers=headers, json=data
    )
    answer = response.json()
    if "choices" in answer:
        return answer["choices"][0]["text"]
    else:
        print("Possible error: query returned:", answer)
    return response.json().get("choices", [{"text": ""}])[0]["text"]


import uuid

from databases import Database

MIGRATION_VERSION_TABLE = "mochaver"

# Maybe rewrite all the await and db stuff with to not be async??


async def table_exists(db: Database, table_name: str):
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name;"
    result = await db.fetch_one(query, values={"table_name": table_name})
    return result is not None


async def get_migration_version(db: Database):
    if await table_exists(db, MIGRATION_VERSION_TABLE):
        version_query = f"SELECT version FROM {MIGRATION_VERSION_TABLE};"
        (result,) = await db.fetch_one(version_query)
        return result
    return None


async def set_version(db: Database, version: int):
    query = f"UPDATE {MIGRATION_VERSION_TABLE} SET version = :version;"
    await db.execute(query, values={"version": version})


MIGRATIONS = {}


async def setup_database(db: Database):
    async with db.transaction():
        # check if table exists for "migration_version"
        migration_version = await get_migration_version(db)
        for _, migration in sorted(MIGRATIONS.items(), key=lambda x: x[0]):
            await migration(db, migration_version)


def migration(version: int):
    def decorator(func):
        async def run_migration(db: Database, db_version: int):
            if db_version is None or db_version < version:
                print("Running migration", version)
                await func(db)
                await set_version(db, version)

        MIGRATIONS[version] = run_migration
        return run_migration

    return decorator


@migration(0)
async def migration_0(db: Database):
    create_migration_table = f"""
        CREATE TABLE {MIGRATION_VERSION_TABLE} (
            version INTEGER NOT NULL PRIMARY KEY
        ) WITHOUT ROWID;
    """
    await db.execute(create_migration_table)
    await db.execute(f"INSERT INTO {MIGRATION_VERSION_TABLE} (version) VALUES (0);")


@migration(1)
async def migration_1(db: Database):
    queries = [
        """
        CREATE TABLE promptHistory (
            id TEXT NOT NULL PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            prompt_name TEXT NOT NULL,
            inputs TEXT NOT NULL,
            response TEXT NOT NULL,
            duration REAL NOT NULL,
            timestamp TEXT NOT NULL
        ) WITHOUT ROWID;
        """,
    ]
    for query in queries:
        await db.execute(query)


async def record_prompt(
    db: Database, prompt_id, prompt_name, inputs, response, duration
):
    query = """
        INSERT INTO promptHistory (id, prompt_id, prompt_name, inputs, response, duration, timestamp)
        VALUES (:id, :prompt_id, :prompt_name, :inputs, :response, :duration, datetime());
    """
    await db.execute(
        query,
        values={
            "id": str(uuid.uuid4()),
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "inputs": json.dumps(inputs),
            "response": response,
            "duration": duration,
        },
    )


def print_prompt_json(prompt_id, prompt_name, inputs, response, duration):
    print(
        json.dumps(
            {
                "id": str(uuid.uuid4()),
                "prompt_id": prompt_id,
                "prompt_name": prompt_name,
                "inputs": json.dumps(inputs),  # not sure, should this dump here?
                "response": response,
                "duration": duration,
            }
        )
    )


async def get_prompts(db: Database, prompt_name: str, n=5):
    query = f"""
        SELECT inputs, response FROM promptHistory
        WHERE prompt_name = :prompt_name
        ORDER BY timestamp DESC
        LIMIT {n};
    """
    result = await db.fetch_all(query, values={"prompt_name": prompt_name})
    return [(json.loads(inputs), response) for inputs, response in result]


database = Database(PM_SETTINGS['LOCAL_HISTORY_DB'])
asyncio.create_task(setup_database(database))

class Prompt:
    # prompts are functions that take in inputs and output strings
    def __init__(self, name, function=None):
        self.name = name
        self.function = function

    def execute(self, *args, **kwargs):
        if self.function is None:
            raise NotImplementedError("Must implement function")
        return self.function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        st = time.time()
        response = self.execute(*args, **kwargs)
        et = time.time()
        if PM_SETTINGS['VERBOSE']:
            print_prompt_json(
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        asyncio.create_task(
            record_prompt(
                database,
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        return md5(inspect.getsource(self.function).encode("utf-8")).hexdigest()


# https://zetcode.com/python/jinja/
class GPT3Prompt(Prompt):
    def __init__(self, name, prompt_template_string, temperature=0, stop=None):
        super().__init__(name)
        self.prompt_template_string = prompt_template_string
        self.prompt_template = env.from_string(prompt_template_string)
        self.stop = stop
        self.temperature = temperature

    def get_named_args(self):
        return meta.find_undeclared_variables(env.parse(self.prompt_template_string))

    def get_prompt(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(**{n: a for n, a in  zip(self.get_named_args(), args)})
        return self.prompt_template.render(**kwargs)

    def execute(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        response = get_gpt3_response(prompt, self.temperature, self.stop)
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        return md5(self.prompt_template_string.encode("utf-8")).hexdigest()


isSQLyn = GPT3Prompt(
    "isSQLyn",
    """
The following is a conversation with a data expert (datAI). 
In order to respond to the prompt, should the datAI use SQL or just say something friendly in response?
The users intent: {{ userIntent }}
===
{{ conversation }}
===
The datAI should use SQL to answer the question (y/n):""",
)


sqlExecutor = GPT3Prompt(
    "sqlExecutor",
    """
The following is a conversation with a data expert (datAI). In order to be accurate, datAI would like to use SQL. 
There is a sqlite database (nba.sql)
UserIntent: {{userIntent}}
---
|| Database Context ||
{{ database_context }}
===
{{ conversation }}
===
{% if previousAttempt %}
The AI has tried already but failed: Previous Query and response
{{ previousAttempt }}
---
New Attempt that is different
{% endif %}
The SQLite query that datAI would like to execute is:
```""",
    stop="```",
)

gotAValidSQLResponse = GPT3Prompt(
    "gotAValidSQLResponse",
    """
===
{{ conversation }}
===
In order to answer a question (above), a data expert (datAI) executed the following SQL query:
{{ sql }}
And the result of the execution was:
{{ result }}
===
Did the query execute successfully (y/n):""",
)

composeDataDrivenAnswer = GPT3Prompt(
    "composeDataDrivenAnswer",
    """"
The following is a conversation with a data expert (datAI). In order to be accurate, the data expert used a database (queried with SQL).
===
{{ conversation }}
===
The query and response:
{{ sql }}
Result:
{{ result }}
===
What should datAI say in response to the last message summarizing what it ran and its result (it can just directly copy if that is the best answer)?
""",
)

neededHelpButStillFriendly = GPT3Prompt(
    "neededHelpButStillFriendly",
    """
The following is a conversation with a friendly chatbot (datAI, a data expert), who loves basketball.
The dataAI tried to run some SQL, but failed to get a good response to the question, 
and now wants to ask a clarifying question so that it can query the database better to help the user...
===
{{ conversation }}
datAI:""",
)

friendlyChatbot = GPT3Prompt(
    "friendlyChatbot",
    """
The following is a conversation with a friendly chatbot (datAI, a data expert), who loves basketball.
The response will never contain a data response unless it has proof in the form of executed SQL. 
datAI does not respond with guesses, so will ask questions to clarify the users intent to help it formulate a SQL query.
datAI never repeats itself either.
userIntent: {{ userIntent }}
===
{{ conversation }}
datAI:""",
)

checkIfNeedsSQL = GPT3Prompt(
    "checkIfNeedsSQL",
    """
The following is a conversation with a friendly chatbot (datAI, a data expert), who loves basketball.
Does the following statement contain any attempt at factual information that would exist in a database?

Statement: {{ statement }}
===
y/n:""",
)

getUserIntent = GPT3Prompt(
    "getUserIntent",
    """
The following is a conversation with a friendly chatbot (datAI, a data expert).
===
{{ conversation }}
===
What is the users intent in this conversation?
""",
)


def starts_with_y(string_data):
    return string_data.strip().lower()[:1] == "y"


def execute_sql(sql, response_limit=500, db_path=None):
    db_path = db_path or PM_SETTINGS['DB_PATH']
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(sql)
        res = str(c.fetchall())[:response_limit]
    except Exception as e:
        res = f"Error executing SQL {e}"
    return res


exec_sql = Prompt("exec_sql", execute_sql)


def get_database_context(db_path=None):
    db_path = db_path or PM_SETTINGS['DB_PATH']
    # Could use this style to wrap rather than asyncio above for the schemas stuff.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # execute a sql query to get all the tables and the columns in the tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    outputschema = ""
    for (table,) in c.fetchall():
        outputschema += table + "("
        c.execute(f"PRAGMA table_info({table})")
        for _, column, dtype, *_ in c.fetchall():
            outputschema += f" {column} "
        outputschema += ") \n"
        output = c.execute(f"select * from {table} limit 1").fetchall()
        if len(output) == 0:
            outputschema += "Empty Table"
        else:
            outputschema += " ".join([str(x) for x in output[0]])
        outputschema += "\n"
    database_context = (
        outputschema + "Example Query (Russell Westbrook's total Triple-Doubles)\n"
    )
    database_context += """
    SELECT SUM(td3) 
    FROM player_game_log 
    LEFT JOIN player ON player.player_id = player_game_log.player_id 
    WHERE player.player_name = 'Russell Westbrook';
    =========
    """
    return database_context


get_db_context = Prompt("get_db_context", get_database_context)


def example_promptchain(conversation, database_context=None):
    if database_context is None:
        database_context = get_db_context()
    userIntent = getUserIntent(conversation=conversation)
    shouldUseSQL = isSQLyn(conversation=conversation, userIntent=userIntent)
    answer = friendlyChatbot(userIntent=userIntent, conversation=conversation)
    needsSql = checkIfNeedsSQL(statement=answer)
    trials = 0
    if starts_with_y(shouldUseSQL) or starts_with_y(needsSql):
        previousAttempt = None
        while trials < 3:
            sqlresponse = sqlExecutor(
                userIntent=userIntent,
                conversation=conversation,
                previousAttempt=previousAttempt,
                database_context=database_context,
            )
            res = exec_sql(sqlresponse)
            validSQL = gotAValidSQLResponse(
                conversation=conversation, sql=sqlresponse, result=res
            )
            if starts_with_y(validSQL):
                return composeDataDrivenAnswer(
                    conversation=conversation, sql=sqlresponse, result=res
                )
            trials += 1
            previousAttempt = f"```{sqlresponse}```\nResult:{res[:100]}\n"
        return neededHelpButStillFriendly(conversation=conversation)
    return answer


epc = Prompt("example_promptchain", example_promptchain)
