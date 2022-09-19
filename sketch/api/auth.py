from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request, Response, WebSocket, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security.oauth2 import (
    HTTP_401_UNAUTHORIZED,
    OAuth2,
    OAuthFlowsModel,
    Optional,
    get_authorization_scheme_param,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from . import data
from .deps import *

# TODO: consider throwing this whole style away, use a tool.

# TODO: put this into an env var or something (env or db? which is better?)
SECRET_KEY = "b684803ea47c9d287a18559418fd531f6e640a5c3f714f7b10b8a0904f7a3b85"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    is_token: bool = False


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class OAuth2PasswordBearerWithCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: str = None,
        scopes: dict = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            authorization: str = request.cookies.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None

        return param


class OAuth2PasswordBearerWithCookieWebsocket(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: str = None,
        scopes: dict = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, websocket: WebSocket) -> Optional[str]:
        authorization = websocket.cookies.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None

        return param


oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="token")
oauth2_scheme_websocket = OAuth2PasswordBearerWithCookieWebsocket(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user_from_db(database, username: str):
    try:
        username, name, email, hashed_password = await data.get_user(database, username)
    except:
        return None
    return UserInDB(
        username=username, email=email, full_name=name, hashed_password=hashed_password
    )


async def authenticate_user(database, username: str, password: str):
    user = await get_user_from_db(database, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_user_from_token(token):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # TODO: fix this to use a database with a function like the get_user_from_db does
    try:
        api_token = await data.get_apikey(database, token)
    except:
        api_token = None
    if api_token:
        username, expires_at = api_token
        user = await get_user_from_db(database, username)
        if not user:
            raise credentials_exception
        user.is_token = True
        return user
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        print(f"Error: {e}")
        raise credentials_exception
    user = await get_user_from_db(database, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_user(token: str = Depends(oauth2_scheme)):
    return await get_user_from_token(token)


async def get_websocket_user(token: str = Depends(oauth2_scheme_websocket)):
    return await get_user_from_token(token)


async def get_token_user(current_user: User = Depends(get_user)):
    if not current_user.is_token:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_browser_user(current_user: User = Depends(get_user)):
    if current_user.is_token:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def logout():
    response = RedirectResponse("/login")
    response.delete_cookie("Authorization")
    return response


async def login_for_access_token(
    response: Response,
    redirect_uri: str = None,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    user = await authenticate_user(database, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    response.set_cookie(
        key="Authorization", value=f"Bearer {access_token}", httponly=True
    )  # set HttpOnly cookie in response
    if redirect_uri:
        response = RedirectResponse(redirect_uri, status_code=status.HTTP_302_FOUND)
    else:
        response = JSONResponse(
            content={"access_token": access_token, "token_type": "bearer"}
        )
    response.set_cookie(
        key="Authorization", value=f"Bearer {access_token}", httponly=True
    )
    return response
