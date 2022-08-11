import json

from databases import Database

from ..core import Portfolio, SketchPad
from ..references import Reference
from . import models

# https://www.encode.io/databases/database_queries/
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
        CREATE TABLE user (
            username TEXT NOT NULL PRIMARY KEY,
            full_name TEXT,
            email TEXT,
            hashed_password TEXT,
            disabled INTEGER
        ) WITHOUT ROWID;
        """,
        """
        CREATE TABLE apikeys (
            key TEXT NOT NULL PRIMARY KEY,
            note TEXT,
            owner_username TEXT,
            FOREIGN KEY(owner_username) REFERENCES user(username)
        ) WITHOUT ROWID;
        """,
        """
        CREATE TABLE _reference (
            id TEXT NOT NULL PRIMARY KEY,
            short_id INTEGER NOT NULL,
            data TEXT NOT NULL,
            type TEXT NOT NULL
        ) WITHOUT ROWID;
        """,
        """
        CREATE TABLE sketchpad (
            id TEXT NOT NULL PRIMARY KEY,
            data TEXT NOT NULL,
            reference_id TEXT,
            upload_at TEXT,
            owner_username TEXT,
            FOREIGN KEY(owner_username) REFERENCES user(username)
        ) WITHOUT ROWID;
        """,
        """
        CREATE TABLE computed_cache (
            id TEXT NOT NULL PRIMARY KEY,
            type TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY(id) REFERENCES sketchpad(id)
        ) WITHOUT ROWID;
        """,
    ]
    for query in queries:
        await db.execute(query)


async def remove_sketchpad(db: Database, user: str, sketchpad: models.SketchPad):
    query = """
        DELETE FROM sketchpad WHERE id = :id AND owner_username = :owner_username;
    """
    await db.execute(
        query,
        values={
            "owner_username": user,
            "id": sketchpad.metadata.id,
        },
    )


async def add_sketchpad(db: Database, user: str, sketchpad: models.SketchPad):
    query = """
        INSERT OR IGNORE INTO sketchpad (id, data, reference_id, upload_at, owner_username)
        VALUES (:id, :data, :reference_id, CURRENT_TIMESTAMP, :owner_username);
    """
    async with db.transaction():
        await ensure_reference(db, sketchpad.reference)
        await db.execute(
            query,
            values={
                "owner_username": user,
                "id": sketchpad.metadata.id,
                "data": sketchpad.json(),
                "reference_id": sketchpad.reference.id,
            },
        )


# Right now, this just returns most recent "fully qualified"
# sketchpads, not a proper full list...
async def get_sketchpads(db: Database, user: str = None):
    query = """
        SELECT
            data
        FROM sketchpad
        WHERE 
            (owner_username = :user) 
            and (NOT reference_id IS NULL)
        GROUP BY
            reference_id, owner_username
        HAVING upload_at = MIN(upload_at)
        ORDER BY upload_at DESC;
    """
    async for d, in db.iterate(query, values={"user": user}):
        yield SketchPad.from_dict(json.loads(d))


async def get_sketchpads_by_id(db: Database, sketchpad_ids, user: str = None):
    query = f"""
        SELECT
            data
        FROM sketchpad
        WHERE 
            (owner_username = :user)
            and 
            (id IN ({','.join([f':{i}' for i in range(len(sketchpad_ids))])}))
    """
    async for d, in db.iterate(
        query, values={"user": user, **{str(i): d for i, d in enumerate(sketchpad_ids)}}
    ):
        yield SketchPad.from_dict(json.loads(d))


# this is the operation to get a collection of sketchpads (matching conditions)
async def get_portfolio(db: Database, user: str = None):
    sketchpads = [d async for d in get_sketchpads(db, user)]
    return Portfolio(sketchpads=sketchpads)


async def ensure_reference(db: Database, reference: Reference):
    query = "INSERT OR IGNORE INTO _reference (id, short_id, data, type) VALUES (:id, :short_id, :data, :type);"
    to_save = reference.dict()
    to_save.update({"data": json.dumps(reference.data)})
    to_save.update(
        {
            "short_id": int.from_bytes(
                bytes.fromhex(reference.id[:16]), "big", signed=True
            )
        }
    )
    await db.execute(query, values=to_save)


async def get_references(db: Database):
    query = """
        SELECT
            short_id,
            id,
            data,
            type
        FROM _reference
    """
    async for short_id, id, data, type in db.iterate(query):
        yield (
            short_id,
            Reference.from_dict({"id": id, "type": type, "data": json.loads(data)}),
        )


async def get_most_recent_sketchpads_by_reference_short_ids(
    db: Database, short_ids, user: str = None
):
    # select short_id, reference_id, sketchpad.id  from sketchpad left join _reference ON reference_id = _reference.id where (owner_username = 'justin') and (short_id IN (8366672332147158452, -799403208509921231)) group by reference_id, owner_username having upload_at = MIN(upload_at) order by upload_at desc;
    query = f"""
        SELECT
            sketchpad.data
        FROM sketchpad
        LEFT JOIN _reference ON reference_id = _reference.id
        WHERE 
            (owner_username = :user)
            and 
            (short_id IN ({','.join([f':{i}' for i in range(len(short_ids))])}))
        GROUP BY
            reference_id, owner_username
        HAVING upload_at = MIN(upload_at)
        ORDER BY upload_at DESC;
    """
    async for d, in db.iterate(
        query,
        values={"user": user, **{str(i): str(d) for i, d in enumerate(short_ids)}},
    ):
        yield SketchPad.from_dict(json.loads(d))
