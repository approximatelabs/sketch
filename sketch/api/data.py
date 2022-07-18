# from databases import Database

# database = Database("sqlite:///test.db")

# @app.on_event("startup")
# async def database_connect():
#     await database.connect()
#     create_table_query = """
#         CREATE TABLE IF NOT EXISTS allportfolios (
#             id str PRIMARY KEY,
#             sketchpad str NOT NULL,
#             owner str NOT NULL
#         );
#         CREATE TABLE IF NOT EXISTS allportfolios (
#             id str PRIMARY KEY,
#             sketchpad str NOT NULL,
#             owner str NOT NULL
#         );
#     """


# @app.on_event("shutdown")
# async def database_disconnect():
#     await database.disconnect()
