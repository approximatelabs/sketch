import os

from ..examples.prompt_machine import *

NBA_DATABASE = "/home/jawaugh/sketch/sketch/examples/datasets/nba_sql.db"
# compare executed results


def get_result(sql):
    conn = sqlite3.connect(NBA_DATABASE)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


async def get_database_context():
    db_path = NBA_DATABASE
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # execute a sql query to get all the tables and the columns in the tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    outputschema = ""
    # outputschema = "Example: table.column* (row1 value) *means join key\n"
    for (table,) in c.fetchall():
        # # # STYLE 2
        # outputschema += f"===== {table} =====\n"
        # c.execute(f"PRAGMA table_info({table})")
        # column_names = [column for _, column, dtype, *_ in c.fetchall()]
        # output = c.execute(f"select * from {table} limit 2").fetchall()
        # for i, column in enumerate(column_names):
        #     if len(output) == 0:
        #         outputschema += f"{column} : -empty-"
        #     elif len(output) == 1:
        #         outputschema += f"{column} : {output[0][i]}"
        #     else:
        #         outputschema += f"{column} : {output[0][i]} | {output[1][i]}"
        #     outputschema += "\n"

        # # STYLE 1
        outputschema += f"{table} ("
        fks = [
            (tab, fro, to)
            for _, _, tab, fro, to, *_ in c.execute(
                f"PRAGMA foreign_key_list({table})"
            ).fetchall()
        ]
        cols_to_star = [co for _, co, _ in fks]
        c.execute(f"PRAGMA table_info({table})")
        columns = [
            column + ("*" if pk or column in cols_to_star else "")
            for _, column, dtype, _, _, pk in c.fetchall()
        ]
        outputschema += "|".join(columns)
        outputschema += ") \n"
        output = c.execute(f"select * from {table} limit 2").fetchall()
        if len(output) == 0:
            outputschema += "Empty Table"
        else:
            for row in output:
                outputschema += "|".join([str(x) for x in row]) + "\n"
        outputschema += (
            "  foreign keys: "
            + ", ".join([f"{fro} -> {to} ({tab})" for tab, fro, to in fks])
            + "\n"
        )
        outputschema += "\n"

        # # STYLE 3
        # fks = [(tab, fro, to) for _, _, tab, fro, to, *_ in c.execute(f"PRAGMA foreign_key_list({table})").fetchall()]
        # cols_to_star = [co for _, co, _ in fks]
        # columns = [column + ("*" if pk or column in cols_to_star else "") for _, column, dtype, _, _, pk in c.execute(f"PRAGMA table_info({table})").fetchall() ]
        # output = c.execute(f"select * from {table} limit 2").fetchall()
        # outputschema += f"== {table} ==\n"
        # for i, column in enumerate(columns):
        #     if len(output) == 0:
        #         outputschema += f"{column} ()"
        #     elif len(output) == 1:
        #         outputschema += f"{column} ({output[0][i]})"
        #     else:
        #         outputschema += f"{column} ({output[0][i]}|{output[1][i]})"
        #     outputschema += "\n"
        # outputschema += "\n"
    return outputschema


get_db_context = asyncPrompt("get_database_context", get_database_context)
correct_sqlite_error = asyncGPT3Edit(
    "correct_sqlite_error",
    "Update the query to fix the error (correct the sql query between the ``` marks): [{{ error }}]",
    model_name="code-davinci-edit-001",
    temperature=0.0,
)

# gpt3_zeroshot_sql = asyncGPT3Prompt("gpt3_zeroshot_sql", """
# Database Context
# {{ db_context }}
# SQLite Query for question [{{ question }}]:
# ```
# """, stop="```", temperature=0.0)

# gpt3_zeroshot_explain_plan = asyncGPT3Prompt("gpt3_zeroshot_explain_plan", """
# Database Context
# {{ db_context }}
# SQLite Query for question [{{ question }}]...
# 1) First we need to consider
# """, stop="```", temperature=1.0, model_name="text-davinci-002")

gpt3_zeroshot_sql_warm = asyncGPT3Prompt(
    "gpt3_zeroshot_sql_warm",
    """
Database Context
{{ db_context }}
SQLite Query for question [{{ question }}]:
```""",
    stop="```",
    temperature=0.4,
    model_name="code-davinci-002",
)


def clean_sql_quick(sql):
    sql = sql.strip()
    sql = sql.replace("\n", "")
    removals = ["sqlite3", "sqlite", "sql", "SQL"]
    for removal in sorted(removals, key=lambda x: len(x), reverse=True):
        if sql.startswith(removal):
            sql = sql[len(removal) :]
    return sql


async def get_sql_for_question_multi(text_in, top_n=5):
    db_context = await get_db_context()
    sqls = await asyncio.gather(
        *[
            gpt3_zeroshot_sql_warm(db_context=db_context, question=text_in)
            for _ in range(top_n)
        ]
    )
    sqls = list(set([clean_sql_quick(sql) for sql in sqls]))

    async def run_sql_for_result(sql):
        trials = 0
        while trials < 4:
            try:
                return (sql, get_result(sql))
            except Exception as e:
                last_prompt = (
                    gpt3_zeroshot_sql_warm.get_prompt(
                        db_context=db_context, question=text_in
                    )
                    + sql
                    + "```"
                )
                response = await correct_sqlite_error(error=str(e), input=last_prompt)
                try:
                    sql = response.split("```")[1]
                    sql = clean_sql_quick(sql)
                except:
                    sql = ""
                trials += 1
        return (sql, None)

    results = await asyncio.gather(*[run_sql_for_result(sql) for sql in sqls])

    answers = {json.dumps(res): set() for _, res in results if res is not None}
    for sql, res in results:
        if res is not None and sql:
            answers[json.dumps(res)].add(sql)
    if len(answers) == 0:
        return "FAILED TO GENERATE SQL"

    top_answers = sorted(answers.values(), key=len, reverse=True)[0]
    return next(iter(top_answers))


get_sql_from_text_multi = asyncPrompt(
    "get_sql_from_text_multi", get_sql_for_question_multi
)


async def get_nba_answer(question):
    """Get the answer to the question from the NBA dataset."""
    sql = await get_sql_from_text_multi(question)
    return get_result(sql)
