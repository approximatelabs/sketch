from databases import Database

# database = Database("sqlite:///test.db")

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


def migration(version: int):
    def decorator(func):
        async def run_migration(db: Database, db_version: int):
            if db_version is None or db_version < version:
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


async def setup_database(db: Database):
    async with db.transaction():
        # check if table exists for "migration_version"
        migration_version = await get_migration_version(db)
        for _, migration in sorted(MIGRATIONS.items(), key=lambda x: x[0]):
            await migration(db, migration_version)
