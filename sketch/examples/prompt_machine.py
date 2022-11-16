import asyncio
import inspect
import json
import os
import sqlite3
import time
import uuid
from hashlib import md5
from re import S

import aiohttp
import requests
import websockets
from dotenv import load_dotenv
from jinja2 import Environment, meta

# This is a very clear work in progress, just trying to move some code into a place where it can be cleaned later, but for now re-used.

# TODO: Completely throw this away and re-build to be actually nice when I know what im doing...

load_dotenv()


PM_SETTINGS = {
    "BOT_TOKEN": "5a3556bb2de44a73ab2e5643cb633a6c",
    "THREAD_ID": "default",
    "DB_PATH": "../../datasets/nba_sql.db",
    "LOCAL_HISTORY_DB": "sqlite+aiosqlite:///promptHistory.db",
    "VERBOSE": False,
    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
}

PM_SETTINGS[
    "uri"
] = f"wss://api.approx.dev/ws/chat?thread_id={PM_SETTINGS['THREAD_ID']}"

env = Environment()


def get_gpt3_completion_reqs(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-002"
):
    if not PM_SETTINGS["openai_api_key"]:
        raise Exception("No OpenAI API key found")
    # print the prompt if verbose mode
    if PM_SETTINGS["VERBOSE"]:
        print(prompt)
    headers = {
        "Authorization": f"Bearer {PM_SETTINGS['openai_api_key']}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": temperature,
        "model": model_name,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
    }
    if stop:
        data["stop"] = stop
    return headers, data


def get_gpt3_edit_reqs(
    instruction, input="", temperature=0.0, model_name="text-davinci-edit-001"
):
    if not PM_SETTINGS["openai_api_key"]:
        raise Exception("No OpenAI API key found")
    # print the prompt if verbose mode
    if PM_SETTINGS["VERBOSE"]:
        print(input, instruction)
    headers = {
        "Authorization": f"Bearer {PM_SETTINGS['openai_api_key']}",
        "Content-Type": "application/json",
    }
    data = {
        "input": input,
        "instruction": instruction,
        "temperature": temperature,
        "model": model_name,
    }
    return headers, data


def get_gpt3_response_choice(answer):
    if "choices" in answer:
        return answer["choices"][0]["text"]
    else:
        print("Possible error: query returned:", answer)
    return answer.get("choices", [{"text": ""}])[0]["text"]


def get_gpt3_response(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-002"
):
    headers, data = get_gpt3_completion_reqs(prompt, temperature, stop, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        response = requests.post(
            "https://api.openai.com/v1/engines/davinci/completions",
            headers=headers,
            data=json.dumps(data),
        )
        answer = response.json()
        if "choices" in answer:
            return get_gpt3_response_choice(answer)
        else:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                print(".", end="")
                time.sleep(trying * 10)
            else:
                print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


async def async_get_gpt3_response(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-002"
):
    headers, data = get_gpt3_completion_reqs(prompt, temperature, stop, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/completions", headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                if "choices" in answer:
                    return get_gpt3_response_choice(answer)
                else:
                    if "Rate limit" in answer.get("error", {}).get("message", ""):
                        print(".", end="")
                        await asyncio.sleep(trying * 10)
                    else:
                        print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


def get_gpt3_edit_response(
    instruction, input="", temperature=0, model_name="text-davinci-edit-001"
):
    headers, data = get_gpt3_edit_reqs(instruction, input, temperature, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        response = requests.post(
            "https://api.openai.com/v1/edits",
            headers=headers,
            data=json.dumps(data),
        )
        answer = response.json()
        if "choices" in answer:
            return get_gpt3_response_choice(answer)
        else:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                print(".", end="")
                time.sleep(trying * 10)
            else:
                print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


async def async_get_gpt3_edit_response(
    instruction, input="", temperature=0.0, model_name="text-davinci-edit-001"
):
    headers, data = get_gpt3_edit_reqs(instruction, input, temperature, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/edits", headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                if "choices" in answer:
                    return get_gpt3_response_choice(answer)
                else:
                    if "Rate limit" in answer.get("error", {}).get("message", ""):
                        print(".", end="")
                        await asyncio.sleep(trying * 10)
                    else:
                        print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


import uuid

from databases import Database

MIGRATION_VERSION_TABLE = "mochaver"


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
            response TEXT,
            duration REAL,
            timestamp TEXT NOT NULL,
            promptstack TEXT,
            parent_task_id TEXT
        ) WITHOUT ROWID;
        """,
    ]
    for query in queries:
        await db.execute(query)


async def record_prompt(
    db: Database, prompt_id, prompt_name, inputs, response, duration, stack
):
    query = """
        INSERT INTO promptHistory (id, prompt_id, prompt_name, inputs, response, duration, timestamp, promptstack, parent_task_id)
        VALUES (:id, :prompt_id, :prompt_name, :inputs, :response, :duration, datetime(), :promptstack, :parent_task_id);
    """
    await db.execute(
        query,
        values={
            "id": str(uuid.uuid4()),
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "inputs": json.dumps(inputs),
            "response": json.dumps(response),
            "duration": duration,
            "promptstack": json.dumps(stack["prompts"]),
            "parent_task_id": stack["task_id"],
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
                "response": json.dumps(response),
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


database = Database(PM_SETTINGS["LOCAL_HISTORY_DB"])

# try:
#     loop = asyncio.get_running_loop()
# except RuntimeError:  # 'RuntimeError: There is no current event loop...'
#     loop = None

# if loop and loop.is_running():
#     asyncio.create_task(setup_database(database))
# else:
#     asyncio.run(setup_database(database))


async def get_prompts_for_task_id(task_id: str):
    query = f"""
        SELECT * FROM promptHistory
        WHERE parent_task_id = :task_id
    """
    result = await database.fetch_all(query, values={"task_id": task_id})
    outputdicts = []
    for row in result:
        outputdict = dict(row)
        outputdict["inputs"] = json.loads(outputdict["inputs"])
        outputdict["response"] = json.loads(outputdict["response"])
        outputdict["promptstack"] = json.loads(outputdict["promptstack"])
        outputdicts.append(outputdict)
    return outputdicts


def get_prompt_stack_and_outer_id():
    promptstack = {"prompts": [], "task_id": None}
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                # we are in a class
                if isinstance(selfobj, Prompt):
                    promptstack["prompts"].append(selfobj.id)
                elif getattr(selfobj, "task_id", None):
                    promptstack["task_id"] = selfobj.task_id
    return promptstack


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
        promptstack = get_prompt_stack_and_outer_id()
        st = time.time()
        asyncio.run(
            record_prompt(
                database,
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                None,
                None,
                promptstack,
            )
        )
        print("Entering prompt", self.name, promptstack)
        try:
            response = self.execute(*args, **kwargs)
        except Exception as e:
            response = f"ERROR\n{e}"
        print("Exiting prompt", self.name, promptstack)
        et = time.time()
        if PM_SETTINGS["VERBOSE"]:
            print_prompt_json(
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        asyncio.run(
            record_prompt(
                database,
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
                promptstack,
            )
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        return md5(inspect.getsource(self.function).encode("utf-8")).hexdigest()


class asyncPrompt(Prompt):
    async def __call__(self, *args, **kwargs):
        promptstack = get_prompt_stack_and_outer_id()
        await record_prompt(
            database,
            self.id,
            self.name,
            {"args": args, "kwargs": kwargs},
            None,
            None,
            promptstack,
        )
        st = time.time()
        print("Entering prompt", self.name, promptstack)
        try:
            response = await self.execute(*args, **kwargs)
        except Exception as e:
            response = f"ERROR\n{e}"
        print("Exiting prompt", self.name, promptstack)
        et = time.time()
        if PM_SETTINGS["VERBOSE"]:
            print_prompt_json(
                self.id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        await record_prompt(
            database,
            self.id,
            self.name,
            {"args": args, "kwargs": kwargs},
            response,
            et - st,
            promptstack,
        )
        return response


class GPT3Prompt(Prompt):
    id_keys = ["prompt_template_string", "stop", "temperature", "model_name"]

    def __init__(
        self,
        name,
        prompt_template_string,
        temperature=0.0,
        stop=None,
        model_name="text-davinci-002",
    ):
        super().__init__(name)
        self.prompt_template_string = prompt_template_string
        self.prompt_template = env.from_string(prompt_template_string)
        self.stop = stop
        self.temperature = temperature
        self.model_name = model_name

    def get_named_args(self):
        return meta.find_undeclared_variables(env.parse(self.prompt_template_string))

    def get_prompt(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.prompt_template.render(**kwargs)

    def execute(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        response = get_gpt3_response(
            prompt, self.temperature, self.stop, self.model_name
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        reprstuff = " ".join([str(getattr(self, k)) for k in self.id_keys])
        return md5(reprstuff.encode("utf-8")).hexdigest()


class asyncGPT3Prompt(asyncPrompt, GPT3Prompt):
    async def execute(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        response = await async_get_gpt3_response(
            prompt, self.temperature, self.stop, self.model_name
        )
        return response


class GPT3Edit(Prompt):
    id_keys = [
        "instruction_template_string",
        "temperature",
        "model_name",
    ]

    def __init__(
        self,
        name,
        instruction_template_string,
        temperature=0.0,
        model_name="code-davinci-edit-001",
    ):
        super().__init__(name)
        self.instruction_template_string = instruction_template_string
        self.instruction_template = env.from_string(instruction_template_string)
        self.temperature = temperature
        self.model_name = model_name

    def get_named_args(self):
        return meta.find_undeclared_variables(
            env.parse(self.instruction_template_string)
        )

    def get_instruction(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.instruction_template.render(**kwargs)

    def get_input(self, *args, **kwargs):
        return kwargs.get("input", "")

    def execute(self, *args, **kwargs):
        input = self.get_input(*args, **kwargs)
        instruction = self.get_instruction(*args, **kwargs)
        response = get_gpt3_edit_response(
            instruction,
            input=input,
            temperature=self.temperature,
            model_name=self.model_name,
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        reprstuff = " ".join([str(getattr(self, k)) for k in self.id_keys])
        return md5(reprstuff.encode("utf-8")).hexdigest()


class asyncGPT3Edit(asyncPrompt, GPT3Edit):
    async def execute(self, *args, **kwargs):
        input = self.get_input(*args, **kwargs)
        instruction = self.get_instruction(*args, **kwargs)
        response = await async_get_gpt3_edit_response(
            instruction,
            input=input,
            temperature=self.temperature,
            model_name=self.model_name,
        )
        return response


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
    db_path = db_path or PM_SETTINGS["DB_PATH"]
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
    db_path = db_path or PM_SETTINGS["DB_PATH"]
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
    SELECT SUM(player_game_log.td3) 
    FROM player_game_log 
    LEFT JOIN player ON player.player_id = player_game_log.player_id 
    WHERE player.player_name = 'Russell Westbrook';

    Example Query (Who scored the most points in any single game)

    SELECT player_name, MAX(pts) 
    FROM player_game_log 
    LEFT JOIN player ON player.player_id = player_game_log.player_id 
    GROUP BY player.player_name
    ORDER BY MAX(pts) DESC
    LIMIT 1;


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
