from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def create_engine(database_url: str):
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)
