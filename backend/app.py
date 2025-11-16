import os
import uuid
import json
import logging
import re
from datetime import datetime
from functools import wraps
from typing import List, Optional

import requests
from flask import (
    Flask,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
    session,
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


###############################################################################
# Application configuration                                                   #
###############################################################################

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
VIDEO_DIR = os.path.join(UPLOAD_ROOT, "videos")
BLOG_IMAGE_DIR = os.path.join(UPLOAD_ROOT, "blogs")
MESSAGE_ATTACH_DIR = os.path.join(UPLOAD_ROOT, "messages")
AI_IMAGE_DIR = os.path.join(UPLOAD_ROOT, "ai_images")
CLOUD_PC_STORAGE_DIR = os.path.join(UPLOAD_ROOT, "cloud_pcs")

for path in [UPLOAD_ROOT, VIDEO_DIR, BLOG_IMAGE_DIR, MESSAGE_ATTACH_DIR, AI_IMAGE_DIR, CLOUD_PC_STORAGE_DIR]:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)

DATABASE_PATH = os.environ.get("DATABASE_PATH")
if not DATABASE_PATH:
    external_dir = os.environ.get("DATABASE_DIR")
    google_drive_default = os.path.join(
        os.path.expanduser("~/Library/CloudStorage/GoogleDrive-mridul6275@gurukultheschool.com/My Drive/Friendly Friends App")
    )
    if external_dir:
        os.makedirs(external_dir, exist_ok=True)
        DATABASE_PATH = os.path.join(external_dir, "friendly_friends.db")
    elif os.path.isdir(os.path.dirname(google_drive_default)) or os.path.isdir(google_drive_default):
        os.makedirs(google_drive_default, exist_ok=True)
        DATABASE_PATH = os.path.join(google_drive_default, "friendly_friends.db")
    else:
        default_instance = os.path.join(BASE_DIR, "instance")
        os.makedirs(default_instance, exist_ok=True)
        DATABASE_PATH = os.path.join(default_instance, "friendly_friends.db")
else:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# Allowed origins for CORS (supports exact matches and regex patterns)
# Define this before session cookie config so we can check for ngrok
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.environ.get("FRONTEND_URL", "http://localhost:5173"),
    "https://jerilyn-nonobligated-punningly.ngrok-free.dev",
    re.compile(r'https://.*\.ngrok-free\.dev'),
    re.compile(r'https://.*\.ngrok\.app'),
]

# Session cookie configuration
# Default to non-secure for local development (HTTP)
# Only use secure cookies when explicitly set or when behind HTTPS proxy
session_cookie_mode = os.environ.get("SESSION_COOKIE_SECURE", "auto").lower()
if session_cookie_mode == "true":
    session_cookie_secure = True
    session_cookie_samesite = 'None'
elif session_cookie_mode == "false":
    session_cookie_secure = False
    session_cookie_samesite = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")
else:
    # Auto-detect: only use secure cookies if explicitly in production or behind HTTPS
    if os.environ.get("APP_ENV") == "production" or os.environ.get("FORCE_SECURE_COOKIES") == "1":
        session_cookie_secure = True
        session_cookie_samesite = 'None'
    elif os.environ.get("NGROK") == "true":
        # Explicitly running behind ngrok - use secure cookies
        session_cookie_secure = True
        session_cookie_samesite = 'None'
    else:
        # Default: non-secure for local HTTP development
        session_cookie_secure = False
        session_cookie_samesite = 'Lax'

app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-secret"),
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{DATABASE_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    JSON_SORT_KEYS=False,
    SESSION_COOKIE_NAME=os.environ.get("SESSION_COOKIE_NAME", "ff_session"),
    SESSION_COOKIE_SAMESITE=session_cookie_samesite,
    SESSION_COOKIE_SECURE=session_cookie_secure,
    # Increase max content length for large video uploads (default is 16MB)
    # Set to 20GB (20 * 1024 * 1024 * 1024 bytes)
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024 * 1024)),
)

def cors_origin_check(origin: str) -> bool:
    if not origin:
        return False
    for allowed in ALLOWED_ORIGINS:
        if isinstance(allowed, re.Pattern):
            if allowed.match(origin):
                return True
        elif origin == allowed:
            return True
    return False

# CORS configuration - use a list of allowed origins
# Flask-CORS doesn't support callables for origins, so we need to provide a list
# We'll handle dynamic origin checking in the after_request hook
cors_allowed_origins_list = [
    "http://localhost:5173",
    r"https://.*\.github\.io",
    "http://127.0.0.1:5173",
    "https://jerilyn-nonobligated-punningly.ngrok-free.dev",
    r"https://.*\.ngrok-free\.dev",  # Regex pattern for ngrok-free.dev
    r"https://.*\.ngrok\.app",  # Regex pattern for ngrok.app
]

CORS(
    app,
    supports_credentials=True,
    resources={r"/api/*": {
        "origins": cors_allowed_origins_list,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Range"],
        "expose_headers": ["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
    }},
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friendly-friends-backend")

db = SQLAlchemy(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

###############################################################################
# Database models                                                              #
###############################################################################


class TimestampMixin:
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class User(TimestampMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    videos = relationship("Video", backref="owner", lazy=True)
    blogs = relationship("Blog", backref="owner", lazy=True)
    messages_sent = relationship(
        "Message",
        foreign_keys="Message.sender_id",
        backref="sender",
        lazy=True,
    )
    messages_received = relationship(
        "Message",
        foreign_keys="Message.recipient_id",
        backref="recipient",
        lazy=True,
    )
    paint_docs = relationship("Paint", backref="owner", lazy=True)
    todos = relationship("Todo", backref="owner", lazy=True)
    chats = relationship("AIChat", backref="owner", lazy=True)

    def to_dict(self):
        try:
            # Get user's roles
            user_roles_list = [role.to_dict() for role in self.roles] if self.roles else []
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "is_admin": self.is_admin,
                "roles": user_roles_list,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_seen": (self.last_seen.isoformat() if self.last_seen else None),
            }
        except Exception as e:
            logger.exception(f"Error converting User {self.id} to dict: {e}")
            # Return a safe fallback
            return {
                "id": self.id,
                "username": getattr(self, 'username', 'Unknown'),
                "email": getattr(self, 'email', ''),
                "is_admin": getattr(self, 'is_admin', False),
                "roles": [],
                "created_at": None,
                "last_seen": None,
            }


class Video(TimestampMixin, db.Model):
    __tablename__ = "videos"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    filename = db.Column(db.String(255), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "filename": self.filename,
            "created_at": self.created_at.isoformat(),
        }


class Blog(TimestampMixin, db.Model):
    __tablename__ = "blogs"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=False)
    image_filename = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "body": self.body,
            "image_filename": self.image_filename,
            "created_at": self.created_at.isoformat(),
        }


class Message(TimestampMixin, db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    body = db.Column(db.Text, nullable=True)
    attachment_filename = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "body": self.body,
            "attachment": self.attachment_filename,
            "created_at": self.created_at.isoformat(),
        }


class Paint(TimestampMixin, db.Model):
    __tablename__ = "paints"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    data = db.Column(db.Text, nullable=False)  # JSON payload describing strokes

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "data": json.loads(self.data),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Todo(TimestampMixin, db.Model):
    __tablename__ = "todos"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    is_completed = db.Column(db.Boolean, default=False, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "is_completed": self.is_completed,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AIChat(TimestampMixin, db.Model):
    __tablename__ = "ai_chats"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)

    messages = relationship("AIMessage", backref="chat", cascade="all, delete-orphan")

    def to_dict(self, include_messages: bool = False):
        payload = {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
        }
        if include_messages:
            payload["messages"] = [m.to_dict() for m in self.messages]
        return payload


class AIMessage(TimestampMixin, db.Model):
    __tablename__ = "ai_messages"

    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey("ai_chats.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }


class AITraining(TimestampMixin, db.Model):
    __tablename__ = "ai_training"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    is_public = db.Column(db.Boolean, default=True, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "instructions": self.instructions,
            "created_by": self.created_by,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat(),
        }


# Junction table for many-to-many relationship between Users and Roles
user_roles = db.Table(
    'user_roles',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('role_id', db.Integer, db.ForeignKey('roles.id'), primary_key=True),
)


class Role(TimestampMixin, db.Model):
    __tablename__ = "roles"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    ai_instructions = db.Column(db.Text, nullable=True)  # AI-generated instructions for what happens when role is assigned
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    # Many-to-many relationship with Users
    users = relationship("User", secondary=user_roles, backref="roles")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ai_instructions": self.ai_instructions,
            "created_by": self.created_by,
            "user_count": len(self.users) if self.users else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AIDoc(TimestampMixin, db.Model):
    __tablename__ = "ai_docs"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    prompt = db.Column(db.Text, nullable=True)  # Original prompt used to generate the doc

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "content": self.content,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AIImage(TimestampMixin, db.Model):
    __tablename__ = "ai_images"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=True)  # Original prompt used to generate the image

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "filename": self.filename,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CloudPC(TimestampMixin, db.Model):
    __tablename__ = "cloud_pcs"

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    os_version = db.Column(db.String(50), default="1.0 beta", nullable=False)
    status = db.Column(db.String(50), default="created", nullable=False)
    storage_used_mb = db.Column(db.Integer, default=0, nullable=False)
    open_apps = db.Column(db.Text, nullable=True)  # JSON string of open apps state

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "os_version": self.os_version,
            "status": self.status,
            "storage_used_mb": self.storage_used_mb,
            "open_apps": self.open_apps,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AIApp(TimestampMixin, db.Model):
    __tablename__ = "ai_apps"

    id = db.Column(db.Integer, primary_key=True)
    developer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    code = db.Column(db.Text, nullable=False, default="")  # HTML/CSS/JS code
    is_live = db.Column(db.Boolean, default=False, nullable=False)  # Visible to community
    live_at = db.Column(db.DateTime, nullable=True)  # When it went live

    def to_dict(self):
        return {
            "id": self.id,
            "developer_id": self.developer_id,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "is_live": self.is_live,
            "live_at": self.live_at.isoformat() if self.live_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AIAppChat(TimestampMixin, db.Model):
    __tablename__ = "ai_app_chats"

    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.Integer, db.ForeignKey("ai_apps.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)  # AI response
    is_user_message = db.Column(db.Boolean, default=True, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "message": self.message,
            "response": self.response,
            "is_user_message": self.is_user_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AIAppDownload(TimestampMixin, db.Model):
    __tablename__ = "ai_app_downloads"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    app_id = db.Column(db.Integer, db.ForeignKey("ai_apps.id"), nullable=False)
    
    # Ensure a user can only download an app once
    __table_args__ = (db.UniqueConstraint('user_id', 'app_id', name='unique_user_app_download'),)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "app_id": self.app_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


###############################################################################
# Helper utilities                                                             #
###############################################################################


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            user_id = session.get("user_id")
            if not user_id:
                try:
                    return jsonify({"error": "Authentication required"}), 401
                except:
                    return make_response(json.dumps({"error": "Authentication required"}), 401, {"Content-Type": "application/json"})
            try:
                return fn(*args, **kwargs)
            except Exception as fn_error:
                # If the function itself fails, let it propagate to the global error handler
                raise
        except Exception as e:
            try:
                logger.exception(f"Error in login_required decorator: {e}")
            except:
                pass
            try:
                return jsonify({"error": "Authentication error"}), 500
            except:
                return make_response(json.dumps({"error": "Authentication error"}), 500, {"Content-Type": "application/json"})
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
        user = db.session.get(User, user_id)
        if not user or not user.is_admin:
            return jsonify({"error": "Admin privileges required"}), 403
        return fn(*args, **kwargs)

    return wrapper


def current_user() -> Optional[User]:
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        # Use query.get() instead of db.session.get() for better error handling
        user = db.session.query(User).filter_by(id=user_id).first()
        if not user:
            # User was deleted but session still exists - clear it
            session.clear()
            return None
        return user
    except Exception as e:
        logger.exception(f"Error getting current user (user_id={user_id}): {e}")
        # Clear session on error to prevent infinite loops
        try:
            session.clear()
        except:
            pass
        return None


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def ensure_unique_filename(directory: str, filename: str) -> str:
    filename = secure_filename(filename)
    name, ext = os.path.splitext(filename)
    unique = f"{name}_{uuid.uuid4().hex}{ext}"
    return unique


def allowed_video(filename: str) -> bool:
    return filename.lower().endswith(".mp4")


def ensure_json_request() -> dict:
    """Ensure request has valid JSON data."""
    if not request.is_json:
        from werkzeug.exceptions import BadRequest
        raise BadRequest("Content-Type must be application/json")
    data = request.get_json(silent=True)
    if data is None:
        from werkzeug.exceptions import BadRequest
        raise BadRequest("Invalid JSON payload")
    if not isinstance(data, dict):
        from werkzeug.exceptions import BadRequest
        raise BadRequest("JSON payload must be an object")
    return data


###############################################################################
# AI helper functions                                                          #
###############################################################################


def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def call_openai(messages: List[dict], system_prompt: Optional[str] = None) -> str:
    client = get_openai_client()
    if not client:
        return (
            "AI service is currently unavailable. Please configure OPENAI_API_KEY "
            "to enable AI-powered responses."
        )

    payload = []
    if system_prompt:
        payload.append({"role": "system", "content": system_prompt})
    payload.extend(messages)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=payload,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise


def transform_error_for_user(raw_error: str) -> str:
    """Use AI (if available) to rewrite errors into friendly language."""
    client = get_openai_client()
    if not client:
        return "Something went wrong. Please try again or contact support."

    prompt = (
        "You are a helpful assistant. Rewrite the following backend error "
        "message so that a non-technical user can understand it without panic.\n\n"
        f"Error: {raw_error}"
    )
    try:
        friendly = call_openai([{ "role": "user", "content": prompt }])
        return friendly
    except Exception:  # pragma: no cover - fallback if AI fails mid-call
        return "An unexpected error occurred. Our team has been notified."


def gather_training_context(message: str) -> List[str]:
    """Return training snippets that match keywords in the user's message."""
    keywords = [word.lower() for word in message.split() if len(word) > 3][:5]
    if not keywords:
        return []

    query = db.session.query(AITraining).filter_by(is_public=True)
    snippets = []
    for training in query:
        if any(keyword in training.instructions.lower() for keyword in keywords):
            snippets.append(training.instructions)
    return snippets[:3]


def search_external_sources(message: str) -> List[str]:
    """Search Wikipedia, Reddit, and blogs (stubbed for now)."""
    results = []
    try:
        wikipedia = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{requests.utils.quote(message.strip())}",
            timeout=3,
        )
        if wikipedia.ok:
            data = wikipedia.json()
            extract = data.get("extract")
            if extract:
                results.append(f"Wikipedia: {extract}")
    except Exception:
        pass

    try:
        reddit = requests.get(
            "https://www.reddit.com/search.json",
            params={"q": message, "limit": 1},
            headers={"User-Agent": "FriendlyFriendsBot/1.0"},
            timeout=3,
        )
        if reddit.ok:
            data = reddit.json()
            top = (
                data.get("data", {})
                .get("children", [{}])[0]
                .get("data", {})
                .get("title")
            )
            if top:
                results.append(f"Reddit: {top}")
    except Exception:
        pass

    blog_matches = (
        db.session.query(Blog)
        .filter(Blog.body.ilike(f"%{message}%"))
        .limit(2)
        .all()
    )
    for blog in blog_matches:
        results.append(f"Blog: {blog.title} - {blog.body[:200]}...")

    return results


###############################################################################
# Root & health endpoints                                                      #
###############################################################################


@app.get("/")
def index():
    """Health check endpoint with database status."""
    try:
        # Test database connection
        db.session.execute(text("SELECT 1"))
        db_status = "ok"
        db_error = None
    except Exception as e:
        logger.exception(f"Database health check failed: {e}")
        db_status = "error"
        db_error = str(e)
    
    response_data = {
        "status": "ok",
        "service": "Friendly Friends AI Backend",
        "time": datetime.utcnow().isoformat(),
        "database": {
            "status": db_status,
            "path": DATABASE_PATH,
            "exists": os.path.isfile(DATABASE_PATH) if DATABASE_PATH else False,
        }
    }
    
    if db_error:
        response_data["database"]["error"] = db_error
    
    return jsonify(response_data)


###############################################################################
# Authentication routes                                                        #
###############################################################################


@app.post("/api/register")
def register():
    data = ensure_json_request()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required."}), 400

    if db.session.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first():
        return jsonify({"error": "Username or email already exists."}), 400

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
    )
    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    return jsonify({"ok": True, "message": "Registration successful", "user": user.to_dict()})


@app.post("/api/login")
def login():
    """Login endpoint with comprehensive error handling."""
    # Wrap everything in a try-except to ensure we always return a response
    try:
        # Parse JSON request
        try:
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON payload"}), 400
            if not isinstance(data, dict):
                return jsonify({"error": "JSON payload must be an object"}), 400
        except Exception as json_error:
            try:
                logger.exception(f"JSON parsing error in /api/login: {json_error}")
            except:
                pass
            return jsonify({"error": "Invalid request format"}), 400
        
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return jsonify({"error": "Username and password are required."}), 400

        # Query user from database
        try:
            user = db.session.query(User).filter_by(username=username).first()
        except Exception as db_error:
            try:
                logger.exception(f"Database error in /api/login: {db_error}")
                db.session.rollback()
            except:
                pass
            return jsonify({"error": "Database error. Please try again."}), 500
        
        if not user:
            return jsonify({"error": "Invalid credentials."}), 401
        
        # Verify password
        try:
            if not verify_password(password, user.password_hash):
                return jsonify({"error": "Invalid credentials."}), 401
        except Exception as pwd_error:
            try:
                logger.exception(f"Password verification error in /api/login: {pwd_error}")
            except:
                pass
            return jsonify({"error": "Authentication error. Please try again."}), 500

        # Set session and serialize user
        try:
            session["user_id"] = user.id
            try:
                logger.info(f"User {username} logged in successfully")
            except:
                pass
            
            # Serialize user
            try:
                user_dict = user.to_dict()
            except Exception as dict_error:
                try:
                    logger.exception(f"User serialization error in /api/login: {dict_error}")
                except:
                    pass
                # Still return success but with minimal user info
                user_dict = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_admin": user.is_admin,
                }
            
            return jsonify({"ok": True, "message": "Login successful", "user": user_dict})
        except Exception as session_error:
            try:
                logger.exception(f"Error setting session in /api/login: {session_error}")
            except:
                pass
            return jsonify({"error": "Failed to complete login. Please try again."}), 500
    except Exception as e:
        # Catch-all for any unexpected errors - ensure we always return something
        try:
            logger.exception(f"Unexpected error in /api/login: {e}")
        except:
            pass
        
        # Don't re-raise HTTP exceptions - handle them directly
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            try:
                return jsonify({"error": e.description}), e.code
            except:
                return make_response(json.dumps({"error": str(e)}), e.code, {"Content-Type": "application/json"})
        
        # Return generic error message - ensure we always return something
        try:
            error_msg = str(e) if app.debug else "Login failed. Please try again."
            return jsonify({"error": error_msg}), 500
        except:
            # Last resort - return a simple response
            return make_response(json.dumps({"error": "Login failed. Please try again."}), 500, {"Content-Type": "application/json"})


@app.post("/api/logout")
@login_required
def logout():
    session.pop("user_id", None)
    resp = jsonify({"ok": True, "message": "Logged out"})
    return resp


@app.get("/api/me")
@login_required
def me():
    """Get current user info with comprehensive error handling."""
    # Wrap everything to ensure we always return a response
    try:
        # Get user_id from session first
        user_id = session.get("user_id")
        if not user_id:
            try:
                session.clear()
            except:
                pass
            try:
                return jsonify({"error": "Authentication required"}), 401
            except:
                return make_response(json.dumps({"error": "Authentication required"}), 401, {"Content-Type": "application/json"})
        
        # Try to get user from database
        try:
            user = db.session.query(User).filter_by(id=user_id).first()
        except Exception as db_error:
            try:
                logger.exception(f"Database error in /api/me: {db_error}")
                db.session.rollback()
            except:
                pass
            try:
                return jsonify({"error": "Database error. Please try again."}), 500
            except:
                return make_response(json.dumps({"error": "Database error. Please try again."}), 500, {"Content-Type": "application/json"})
        
        if not user:
            # Session has user_id but user doesn't exist - clear session
            try:
                session.clear()
            except:
                pass
            try:
                return jsonify({"error": "User not found"}), 401
            except:
                return make_response(json.dumps({"error": "User not found"}), 401, {"Content-Type": "application/json"})
        
        # Try to serialize user
        try:
            user_dict = user.to_dict()
            try:
                return jsonify({"ok": True, "user": user_dict})
            except:
                return make_response(json.dumps({"ok": True, "user": user_dict}), 200, {"Content-Type": "application/json"})
        except Exception as dict_error:
            try:
                logger.exception(f"Error serializing user in /api/me: {dict_error}")
            except:
                pass
            # Fallback to minimal user info
            try:
                user_dict = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_admin": user.is_admin,
                }
                try:
                    return jsonify({"ok": True, "user": user_dict})
                except:
                    return make_response(json.dumps({"ok": True, "user": user_dict}), 200, {"Content-Type": "application/json"})
            except:
                try:
                    return jsonify({"error": "Failed to retrieve user information"}), 500
                except:
                    return make_response(json.dumps({"error": "Failed to retrieve user information"}), 500, {"Content-Type": "application/json"})
    except Exception as e:
        # Catch-all - ensure we always return something
        try:
            logger.exception(f"Unexpected error in /api/me: {e}")
        except:
            pass
        try:
            return jsonify({"error": "Failed to retrieve user information"}), 500
        except:
            return make_response(json.dumps({"error": "Failed to retrieve user information"}), 500, {"Content-Type": "application/json"})


###############################################################################
# Member management                                                            #
###############################################################################


@app.get("/api/members")
@login_required
def list_members():
    members = db.session.query(User).order_by(User.created_at.desc()).all()
    return jsonify({"members": [m.to_dict() for m in members]})


@app.post("/api/members")
@admin_required
def create_member():
    data = ensure_json_request()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "") or uuid.uuid4().hex[:10]
    is_admin = bool(data.get("is_admin"))

    if not username or not email:
        return jsonify({"error": "Username and email required"}), 400

    if db.session.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first():
        return jsonify({"error": "Username or email already exists."}), 400

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        is_admin=is_admin,
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Member created", "user": user.to_dict()})


@app.delete("/api/members/<int:user_id>")
@admin_required
def delete_member(user_id: int):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "Member not found"}), 404

    if user.id == session.get("user_id"):
        return jsonify({"error": "Admins cannot delete themselves."}), 400

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "Member deleted"})


###############################################################################
# Video sharing                                                                #
###############################################################################


@app.get("/api/videos")
@login_required
def list_videos():
    videos = db.session.query(Video).order_by(Video.created_at.desc()).all()
    return jsonify({"videos": [v.to_dict() for v in videos]})


@app.post("/api/videos")
@login_required
def upload_video():
    """Upload video with support for large files."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        if "video" not in request.files:
            return jsonify({"error": "Video file is required"}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_video(file.filename):
            return jsonify({"error": "Only MP4 videos are supported."}), 400

        title = request.form.get("title", "Untitled Video")
        description = request.form.get("description")

        # Check file size (if available)
        try:
            # Seek to end to get file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            # Log file size for debugging
            size_mb = file_size / (1024 * 1024)
            logger.info(f"Uploading video: {file.filename}, size: {size_mb:.2f} MB")
            
            # Warn if file is very large (>1GB)
            if file_size > 1024 * 1024 * 1024:
                logger.warning(f"Large video file detected: {size_mb:.2f} MB")
        except Exception as size_error:
            logger.warning(f"Could not determine file size: {size_error}")
            # Continue anyway - file size check is optional

        filename = ensure_unique_filename(VIDEO_DIR, file.filename)
        filepath = os.path.join(VIDEO_DIR, filename)
        
        # Save file with error handling
        try:
            file.save(filepath)
            logger.info(f"Video saved to: {filepath}")
        except Exception as save_error:
            logger.exception(f"Error saving video file: {save_error}")
            return jsonify({"error": f"Failed to save video file: {str(save_error)}"}), 500

        # Create database record
        try:
            video = Video(
                owner_id=user.id,
                title=title,
                description=description,
                filename=filename,
            )
            db.session.add(video)
            db.session.commit()
            logger.info(f"Video record created: ID {video.id}")
        except Exception as db_error:
            logger.exception(f"Error creating video record: {db_error}")
            # Try to clean up the file if database save fails
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            db.session.rollback()
            return jsonify({"error": "Failed to save video metadata. Please try again."}), 500

        return jsonify({"message": "Video uploaded successfully", "video": video.to_dict()})
    except Exception as e:
        logger.exception(f"Unexpected error in upload_video: {e}")
        return jsonify({"error": "Failed to upload video. Please try again."}), 500


@app.get("/api/videos/<int:video_id>")
@login_required
def get_video(video_id: int):
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404
    return jsonify(video.to_dict())


@app.get("/api/videos/<int:video_id>/stream")
@login_required
def stream_video(video_id: int):
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404
    filepath = os.path.join(VIDEO_DIR, video.filename)
    if not os.path.exists(filepath):
        logger.warning(f"Video file missing: {filepath} for video ID {video_id}")
        return jsonify({"error": "Video file missing"}), 404
    
    # Set proper headers for video streaming
    response = send_file(
        filepath,
        mimetype="video/mp4",
        conditional=True,
        as_attachment=False
    )
    # Add CORS headers for video streaming
    origin = request.headers.get('Origin')
    if origin and cors_origin_check(origin):
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Range, Accept-Ranges, Content-Length, Content-Type'
    return response


@app.delete("/api/videos/<int:video_id>")
@login_required
def delete_video(video_id: int):
    user = current_user()
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404
    if video.owner_id != user.id and not user.is_admin:
        return jsonify({"error": "Not allowed"}), 403

    filepath = os.path.join(VIDEO_DIR, video.filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    db.session.delete(video)
    db.session.commit()
    return jsonify({"message": "Video deleted"})


###############################################################################
# Blog creation                                                                #
###############################################################################


@app.get("/api/blogs")
@login_required
def list_blogs():
    blogs = db.session.query(Blog).order_by(Blog.created_at.desc()).all()
    return jsonify({"blogs": [b.to_dict() for b in blogs]})


@app.post("/api/blogs")
@login_required
def create_blog():
    user = current_user()
    title = request.form.get("title", "Untitled Blog").strip()
    body = request.form.get("body", "").strip()
    if not title or not body:
        return jsonify({"error": "Title and body are required"}), 400

    image_filename = None
    if "image" in request.files:
        image = request.files["image"]
        if image.filename:
            image_filename = ensure_unique_filename(BLOG_IMAGE_DIR, image.filename)
            image.save(os.path.join(BLOG_IMAGE_DIR, image_filename))

    blog = Blog(owner_id=user.id, title=title, body=body, image_filename=image_filename)
    db.session.add(blog)
    db.session.commit()
    return jsonify({"message": "Blog created", "blog": blog.to_dict()})


@app.get("/api/blogs/<int:blog_id>")
@login_required
def get_blog(blog_id: int):
    blog = db.session.get(Blog, blog_id)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
    return jsonify(blog.to_dict())


@app.put("/api/blogs/<int:blog_id>")
@login_required
def update_blog(blog_id: int):
    user = current_user()
    blog = db.session.get(Blog, blog_id)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
    if blog.owner_id != user.id and not user.is_admin:
        return jsonify({"error": "Not allowed"}), 403

    data = ensure_json_request()
    blog.title = data.get("title", blog.title)
    blog.body = data.get("body", blog.body)
    db.session.commit()
    return jsonify({"message": "Blog updated", "blog": blog.to_dict()})


@app.delete("/api/blogs/<int:blog_id>")
@login_required
def delete_blog(blog_id: int):
    user = current_user()
    blog = db.session.get(Blog, blog_id)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
    if blog.owner_id != user.id and not user.is_admin:
        return jsonify({"error": "Not allowed"}), 403

    if blog.image_filename:
        img_path = os.path.join(BLOG_IMAGE_DIR, blog.image_filename)
        if os.path.exists(img_path):
            os.remove(img_path)

    db.session.delete(blog)
    db.session.commit()
    return jsonify({"message": "Blog deleted"})


@app.get("/uploads/blogs/<path:filename>")
@login_required
def get_blog_image(filename: str):
    return send_from_directory(BLOG_IMAGE_DIR, filename)


###############################################################################
# Messaging                                                                    #
###############################################################################


@app.post("/api/presence/update")
@login_required
def update_presence():
    try:
        # Get user_id from session (already checked by login_required)
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get user from database
        try:
            user = db.session.query(User).filter_by(id=user_id).first()
        except Exception as db_error:
            logger.exception(f"Database error in /api/presence/update: {db_error}")
            return jsonify({"error": "Database error. Please try again."}), 500
        
        if not user:
            session.clear()
            return jsonify({"error": "User not found"}), 401
        
        # Update presence
        try:
            payload = request.get_json(silent=True) or {}
            status = (payload.get("status") or "online").strip()
            user.last_seen = datetime.utcnow()
            db.session.commit()
            return jsonify({
                "message": "Presence updated",
                "status": status,
                "last_seen": user.last_seen.isoformat(),
            })
        except Exception as update_error:
            logger.exception(f"Error updating presence: {update_error}")
            db.session.rollback()
            return jsonify({"error": "Failed to update presence"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /api/presence/update: {e}")
        return jsonify({"error": "Failed to update presence"}), 500


# Video calls endpoints (stub - video calls feature was cancelled)
@app.get("/api/calls/pending")
@login_required
def get_pending_calls():
    """Get pending video calls (stub - returns no calls)."""
    try:
        # Just verify user is authenticated (already checked by login_required)
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
        
        # Return stub response
        return jsonify({"pending": False, "status": None, "calls": []})
    except Exception as e:
        logger.exception(f"Error in /api/calls/pending: {e}")
        return jsonify({"pending": False, "status": None, "calls": []})


@app.post("/api/calls/<int:call_id>/accept")
@login_required
def accept_call(call_id: int):
    """Accept a video call (stub)."""
    return jsonify({"error": "Video calls feature is not available"}), 404


@app.post("/api/calls/<int:call_id>/reject")
@login_required
def reject_call(call_id: int):
    """Reject a video call (stub)."""
    return jsonify({"ok": True, "message": "Call rejected"})


@app.post("/api/calls/<int:call_id>/end")
@login_required
def end_call(call_id: int):
    """End a video call (stub)."""
    return jsonify({"ok": True, "message": "Call ended"})


def conversation_query(user_id: int, other_user_id: int):
    return (
        db.session.query(Message)
        .filter(
            ((Message.sender_id == user_id) & (Message.recipient_id == other_user_id))
            | ((Message.sender_id == other_user_id) & (Message.recipient_id == user_id))
        )
        .order_by(Message.created_at.asc())
    )


@app.get("/api/messages")
@login_required
def list_recent_conversations():
    user = current_user()
    conversations = (
        db.session.query(Message.recipient_id)
        .filter(Message.sender_id == user.id)
        .union(
            db.session.query(Message.sender_id).filter(Message.recipient_id == user.id)
        )
        .all()
    )
    partner_ids = {row[0] for row in conversations if row[0] != user.id}
    partners = db.session.query(User).filter(User.id.in_(partner_ids)).all()
    return jsonify({"partners": [p.to_dict() for p in partners]})


@app.get("/api/messages/<username>")
@login_required
def get_conversation(username: str):
    user = current_user()
    partner = db.session.query(User).filter_by(username=username).first()
    if not partner:
        return jsonify({"error": "User not found"}), 404

    messages = conversation_query(user.id, partner.id).all()
    return jsonify({
        "partner": partner.to_dict(),
        "messages": [m.to_dict() for m in messages],
    })


@app.post("/api/messages")
@login_required
def send_message():
    user = current_user()
    recipient_username = None
    body = None

    if request.content_type and request.content_type.startswith("multipart"):
        recipient_username = (request.form.get("recipient") or "").strip()
        body = request.form.get("body")
    else:
        payload = request.get_json(silent=True) or {}
        recipient_username = (payload.get("recipient") or "").strip()
        body = payload.get("body")

    if not recipient_username:
        return jsonify({"error": "Recipient username required"}), 400

    recipient = db.session.query(User).filter_by(username=recipient_username).first()
    if not recipient:
        return jsonify({"error": "Recipient not found"}), 404

    if not body and "attachment" not in request.files:
        return jsonify({"error": "A message body or attachment is required"}), 400

    attachment_filename = None
    if "attachment" in request.files:
        attachment = request.files["attachment"]
        if attachment.filename:
            attachment_filename = ensure_unique_filename(
                MESSAGE_ATTACH_DIR, attachment.filename
            )
            attachment.save(os.path.join(MESSAGE_ATTACH_DIR, attachment_filename))

    message = Message(
        sender_id=user.id,
        recipient_id=recipient.id,
        body=body,
        attachment_filename=attachment_filename,
    )
    db.session.add(message)
    db.session.commit()

    return jsonify({"message": "Message sent", "data": message.to_dict()})


@app.get("/uploads/messages/<path:filename>")
@login_required
def get_message_attachment(filename: str):
    return send_from_directory(MESSAGE_ATTACH_DIR, filename)


###############################################################################
# Paint                                                                        #
###############################################################################


@app.get("/api/paint")
@login_required
def list_paint_docs():
    user = current_user()
    docs = db.session.query(Paint).filter_by(owner_id=user.id).all()
    return jsonify({"paintings": [doc.to_dict() for doc in docs]})


@app.post("/api/paint")
@login_required
def save_paint_doc():
    user = current_user()
    data = ensure_json_request()
    name = data.get("name", "Untitled Canvas").strip()
    strokes = data.get("data")
    if strokes is None:
        return jsonify({"error": "Paint data is required"}), 400

    paint_id = data.get("id")
    if paint_id:
        paint = db.session.get(Paint, paint_id)
        if not paint or paint.owner_id != user.id:
            return jsonify({"error": "Paint not found"}), 404
        paint.name = name
        paint.data = json.dumps(strokes)
    else:
        paint = Paint(owner_id=user.id, name=name, data=json.dumps(strokes))
        db.session.add(paint)

    db.session.commit()
    return jsonify({"message": "Paint saved", "paint": paint.to_dict()})


@app.delete("/api/paint/<int:paint_id>")
@login_required
def delete_paint_doc(paint_id: int):
    user = current_user()
    paint = db.session.get(Paint, paint_id)
    if not paint or paint.owner_id != user.id:
        return jsonify({"error": "Paint not found"}), 404
    db.session.delete(paint)
    db.session.commit()
    return jsonify({"message": "Paint deleted"})


###############################################################################
# Todos                                                                        #
###############################################################################


@app.get("/api/todos")
@login_required
def list_todos():
    user = current_user()
    todos = db.session.query(Todo).filter_by(owner_id=user.id).order_by(Todo.created_at).all()
    return jsonify({"todos": [todo.to_dict() for todo in todos]})


@app.post("/api/todos")
@login_required
def create_todo():
    user = current_user()
    data = ensure_json_request()
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400

    todo = Todo(owner_id=user.id, title=title)
    db.session.add(todo)
    db.session.commit()
    return jsonify({"message": "Todo created", "todo": todo.to_dict()})


@app.put("/api/todos/<int:todo_id>")
@login_required
def update_todo(todo_id: int):
    user = current_user()
    todo = db.session.get(Todo, todo_id)
    if not todo or todo.owner_id != user.id:
        return jsonify({"error": "Todo not found"}), 404

    data = ensure_json_request()
    todo.title = data.get("title", todo.title)
    todo.is_completed = bool(data.get("is_completed", todo.is_completed))
    db.session.commit()
    return jsonify({"message": "Todo updated", "todo": todo.to_dict()})


@app.delete("/api/todos/<int:todo_id>")
@login_required
def delete_todo(todo_id: int):
    user = current_user()
    todo = db.session.get(Todo, todo_id)
    if not todo or todo.owner_id != user.id:
        return jsonify({"error": "Todo not found"}), 404
    db.session.delete(todo)
    db.session.commit()
    return jsonify({"message": "Todo deleted"})


###############################################################################
# AI Chat & Docs                                                               #
###############################################################################


@app.get("/api/ai/chats")
@login_required
def list_ai_chats():
    user = current_user()
    chats = db.session.query(AIChat).filter_by(owner_id=user.id).order_by(AIChat.created_at.desc()).all()
    return jsonify({"chats": [chat.to_dict() for chat in chats]})


@app.get("/api/ai/chats/<int:chat_id>")
@login_required
def get_ai_chat(chat_id: int):
    user = current_user()
    chat = db.session.get(AIChat, chat_id)
    if not chat or chat.owner_id != user.id:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat.to_dict(include_messages=True))


@app.get("/api/ai/chats/<int:chat_id>/messages")
@login_required
def get_ai_chat_messages(chat_id: int):
    """Get messages for a specific chat."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get chat and verify ownership
        try:
            chat = db.session.query(AIChat).filter_by(id=chat_id, owner_id=user.id).first()
        except Exception as db_error:
            logger.exception(f"Database error in /api/ai/chats/{chat_id}/messages: {db_error}")
            db.session.rollback()
            return jsonify({"error": "Database error. Please try again."}), 500
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Get messages for this chat
        try:
            messages = db.session.query(AIMessage).filter_by(chat_id=chat_id).order_by(AIMessage.created_at.asc()).all()
            messages_list = [m.to_dict() for m in messages]
        except Exception as msg_error:
            logger.exception(f"Error loading messages: {msg_error}")
            db.session.rollback()
            return jsonify({"error": "Failed to load messages"}), 500
        
        return jsonify({
            "chat": chat.to_dict(),
            "messages": messages_list
        })
    except Exception as e:
        logger.exception(f"Unexpected error in /api/ai/chats/{chat_id}/messages: {e}")
        return jsonify({"error": "Failed to load chat messages"}), 500


@app.delete("/api/ai/chats/<int:chat_id>")
@login_required
def delete_ai_chat(chat_id: int):
    """Delete an AI chat and all its messages."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get chat and verify ownership
        try:
            chat = db.session.query(AIChat).filter_by(id=chat_id, owner_id=user.id).first()
        except Exception as db_error:
            logger.exception(f"Database error in /api/ai/chats/{chat_id} DELETE: {db_error}")
            db.session.rollback()
            return jsonify({"error": "Database error. Please try again."}), 500
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Delete the chat (messages will be cascade deleted due to relationship)
        try:
            db.session.delete(chat)
            db.session.commit()
            logger.info(f"User {user.id} deleted AI chat {chat_id}")
            return jsonify({"message": "Chat deleted successfully"})
        except Exception as delete_error:
            logger.exception(f"Error deleting chat: {delete_error}")
            db.session.rollback()
            return jsonify({"error": "Failed to delete chat"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /api/ai/chats/{chat_id} DELETE: {e}")
        return jsonify({"error": "Failed to delete chat"}), 500


@app.post("/api/ai/chat")
@login_required
def send_ai_message():
    user = current_user()
    data = ensure_json_request()
    message_content = data.get("message", "").strip()
    chat_id = data.get("chat_id")
    if not message_content:
        return jsonify({"error": "Message is required"}), 400

    if chat_id:
        chat = db.session.get(AIChat, chat_id)
        if not chat or chat.owner_id != user.id:
            return jsonify({"error": "Chat not found"}), 404
    else:
        chat = AIChat(owner_id=user.id, title=data.get("title", "Untitled Chat"))
        db.session.add(chat)
        db.session.flush()  # assign id before commit

    user_msg = AIMessage(chat_id=chat.id, role="user", content=message_content)
    db.session.add(user_msg)

    training_snippets = gather_training_context(message_content)
    search_snippets = search_external_sources(message_content)

    system_prompt = "You are Friendly Friends AI, a warm, concise companion."
    messages_payload = [
        {"role": "user", "content": message_content},
    ]
    context_parts = []
    if training_snippets:
        context_parts.append("Training:\n" + "\n".join(training_snippets))
    if search_snippets:
        context_parts.append("Search Results:\n" + "\n".join(search_snippets))

    if context_parts:
        messages_payload.insert(0, {"role": "assistant", "content": "\n\n".join(context_parts)})

    ai_response = call_openai(messages_payload, system_prompt=system_prompt)

    assistant_msg = AIMessage(chat_id=chat.id, role="assistant", content=ai_response)
    db.session.add(assistant_msg)
    db.session.commit()

    return jsonify({
        "chat": chat.to_dict(),
        "messages": [user_msg.to_dict(), assistant_msg.to_dict()],
    })


@app.get("/api/ai/docs")
@login_required
def list_ai_docs():
    user = current_user()
    docs = db.session.query(AIDoc).filter_by(owner_id=user.id).order_by(AIDoc.updated_at.desc()).all()
    return jsonify({"docs": [d.to_dict() for d in docs]})

@app.post("/api/ai/docs")
@login_required
def generate_ai_doc():
    user = current_user()
    data = ensure_json_request()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    instructions = (
        "You are Friendly Friends AI, create a helpful document for the user."
        " Keep it concise, positive, and actionable."
    )
    doc_text = call_openai([{ "role": "user", "content": prompt }], system_prompt=instructions)
    
    # Generate title from prompt (first 50 chars)
    title = prompt[:50] + ("..." if len(prompt) > 50 else "")
    
    # Save the document
    doc = AIDoc(
        owner_id=user.id,
        title=title,
        content=doc_text,
        prompt=prompt,
    )
    db.session.add(doc)
    db.session.commit()
    
    return jsonify({"message": "Document generated", "doc": doc.to_dict()})

@app.get("/api/ai/docs/<int:doc_id>")
@login_required
def get_ai_doc(doc_id: int):
    user = current_user()
    doc = db.session.get(AIDoc, doc_id)
    if not doc or doc.owner_id != user.id:
        return jsonify({"error": "Document not found"}), 404
    return jsonify(doc.to_dict())

@app.put("/api/ai/docs/<int:doc_id>")
@login_required
def update_ai_doc(doc_id: int):
    user = current_user()
    doc = db.session.get(AIDoc, doc_id)
    if not doc or doc.owner_id != user.id:
        return jsonify({"error": "Document not found"}), 404
    
    data = ensure_json_request()
    if "title" in data:
        doc.title = data["title"].strip()
    if "content" in data:
        doc.content = data["content"]
    
    db.session.commit()
    return jsonify({"message": "Document updated", "doc": doc.to_dict()})

@app.delete("/api/ai/docs/<int:doc_id>")
@login_required
def delete_ai_doc(doc_id: int):
    user = current_user()
    doc = db.session.get(AIDoc, doc_id)
    if not doc or doc.owner_id != user.id:
        return jsonify({"error": "Document not found"}), 404
    
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"message": "Document deleted"})


###############################################################################
# AI Image Generation                                                         #
###############################################################################


@app.get("/api/ai/images")
@login_required
def list_ai_images():
    user = current_user()
    images = db.session.query(AIImage).filter_by(owner_id=user.id).order_by(AIImage.created_at.desc()).all()
    return jsonify({"images": [img.to_dict() for img in images]})


@app.post("/api/ai/images")
@login_required
def generate_ai_image():
    user = current_user()
    data = ensure_json_request()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    client = get_openai_client()
    if not client:
        return jsonify({"error": "AI service is currently unavailable. Please configure OPENAI_API_KEY"}), 503

    try:
        # First, use GPT to analyze the prompt and create a detailed, structured prompt
        # This helps identify all components needed and creates a better prompt for DALL-E
        try:
            analysis_prompt = f"""Analyze this image generation request and create a detailed, structured prompt for DALL-E 3.

Original request: "{prompt}"

Break down the request into components:
1. Main subject(s)
2. Background/environment
3. Style/aesthetic (photorealistic, standard photography style)
4. Lighting conditions
5. Composition details
6. Color palette
7. Any specific elements that need to be included

Create a comprehensive prompt that:
- Uses photorealistic, standard photography style (not artistic/textured)
- Clearly describes all visual elements
- Ensures all components are properly combined
- Uses natural, realistic lighting
- Maintains professional photography quality

Respond ONLY with the enhanced prompt, nothing else."""

            # Get enhanced prompt from GPT
            enhanced_prompt = call_openai(
                [{ "role": "user", "content": analysis_prompt }],
                system_prompt="You are an expert at creating detailed image generation prompts. Create clear, comprehensive prompts that result in photorealistic, standard-quality images."
            )
            
            # Use the enhanced prompt for DALL-E
            final_prompt = enhanced_prompt.strip() if enhanced_prompt else prompt
        except Exception as e:
            # If prompt enhancement fails, use original prompt
            logging.warning(f"Failed to enhance prompt, using original: {e}")
            final_prompt = prompt
        
        # Generate image using DALL-E with the enhanced prompt
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            n=1,
            size="1024x1024",
            quality="standard",
            response_format="url"
        )
        
        image_url = response.data[0].url
        
        # Download the image
        img_response = requests.get(image_url)
        img_response.raise_for_status()
        
        # Generate filename
        title = prompt[:50] + ("..." if len(prompt) > 50 else "")
        sanitized_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        filename = f"{sanitized_title}_{uuid.uuid4().hex[:8]}.jpeg"
        filepath = os.path.join(AI_IMAGE_DIR, filename)
        
        # Save as JPEG
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_response.content))
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        img.save(filepath, 'JPEG', quality=95)
        
        # Save to database (store both original and enhanced prompt)
        ai_image = AIImage(
            owner_id=user.id,
            title=title,
            filename=filename,
            prompt=f"Original: {prompt}\nEnhanced: {final_prompt}",  # Store both prompts
        )
        db.session.add(ai_image)
        db.session.commit()
        
        return jsonify({"message": "Image generated", "image": ai_image.to_dict()})
    except Exception as e:
        logging.error(f"Error generating image: {e}", exc_info=True)
        # Try to get friendly error, but don't fail if that also fails
        try:
            friendly_error = call_openai(
                [{ "role": "user", "content": f"Transform this error into a friendly user message: {str(e)}" }],
                system_prompt="You are Friendly Friends AI. Transform technical errors into friendly, helpful messages."
            )
        except:
            # Fallback to a generic friendly message
            friendly_error = f"Failed to generate image. Please check your prompt and try again. Error: {str(e)[:100]}"
        return jsonify({"error": friendly_error}), 500


@app.get("/api/ai/images/<int:image_id>")
@login_required
def get_ai_image(image_id: int):
    user = current_user()
    ai_image = db.session.get(AIImage, image_id)
    if not ai_image or ai_image.owner_id != user.id:
        return jsonify({"error": "Image not found"}), 404
    return jsonify(ai_image.to_dict())


@app.delete("/api/ai/images/<int:image_id>")
@login_required
def delete_ai_image(image_id: int):
    user = current_user()
    ai_image = db.session.get(AIImage, image_id)
    if not ai_image or ai_image.owner_id != user.id:
        return jsonify({"error": "Image not found"}), 404
    
    # Delete file
    filepath = os.path.join(AI_IMAGE_DIR, ai_image.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    db.session.delete(ai_image)
    db.session.commit()
    return jsonify({"message": "Image deleted"})


@app.get("/uploads/ai_images/<filename>")
@login_required
def get_ai_image_file(filename: str):
    return send_from_directory(AI_IMAGE_DIR, filename)


###############################################################################
# AI Training (admin only)                                                     #
###############################################################################


@app.get("/api/ai/training")
@admin_required
def list_training_items():
    trainings = db.session.query(AITraining).order_by(AITraining.created_at.desc()).all()
    return jsonify({"training": [t.to_dict() for t in trainings]})


@app.post("/api/ai/training")
@admin_required
def create_training_item():
    user = current_user()
    data = ensure_json_request()
    title = data.get("title", "").strip()
    instructions = data.get("instructions", "").strip()
    is_public = bool(data.get("is_public", True))

    if not title or not instructions:
        return jsonify({"error": "Title and instructions are required"}), 400

    training = AITraining(
        title=title,
        instructions=instructions,
        created_by=user.id,
        is_public=is_public,
    )
    db.session.add(training)
    db.session.commit()
    return jsonify({"message": "Training item added", "training": training.to_dict()})


@app.delete("/api/ai/training/<int:training_id>")
@admin_required
def delete_training_item(training_id: int):
    training = db.session.get(AITraining, training_id)
    if not training:
        return jsonify({"error": "Training item not found"}), 404
    db.session.delete(training)
    db.session.commit()
    return jsonify({"message": "Training item deleted"})


###############################################################################
# Roles Management                                                             #
###############################################################################


@app.get("/api/roles")
@login_required
def list_roles():
    """Get all roles - anyone can view roles."""
    try:
        roles = db.session.query(Role).order_by(Role.created_at.desc()).all()
        return jsonify({"roles": [role.to_dict() for role in roles]})
    except Exception as e:
        logger.exception(f"Error listing roles: {e}")
        return jsonify({"error": "Failed to load roles"}), 500


@app.post("/api/roles")
@login_required
def create_role():
    """Create a new role - anyone can create roles."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        data = ensure_json_request()
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        ai_instructions = data.get("ai_instructions", "").strip()
        
        if not name:
            return jsonify({"error": "Role name is required"}), 400
        
        # Check if role with same name already exists
        existing = db.session.query(Role).filter_by(name=name).first()
        if existing:
            return jsonify({"error": "A role with this name already exists"}), 400
        
        role = Role(
            name=name,
            description=description,
            ai_instructions=ai_instructions,
            created_by=user.id
        )
        db.session.add(role)
        db.session.commit()
        
        logger.info(f"User {user.id} created role: {name}")
        return jsonify({"message": "Role created successfully", "role": role.to_dict()})
    except Exception as e:
        logger.exception(f"Error creating role: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to create role"}), 500


@app.post("/api/roles/<int:role_id>/assign")
@admin_required
def assign_role(role_id: int):
    """Assign a role to a user - admin only."""
    try:
        user = current_user()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        data = ensure_json_request()
        user_id = data.get("user_id")
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Get role and target user
        role = db.session.get(Role, role_id)
        if not role:
            return jsonify({"error": "Role not found"}), 404
        
        target_user = db.session.get(User, user_id)
        if not target_user:
            return jsonify({"error": "User not found"}), 404
        
        # Check if user already has this role
        if role in target_user.roles:
            return jsonify({"error": "User already has this role"}), 400
        
        # Assign role
        target_user.roles.append(role)
        db.session.commit()
        
        # Refresh the user object to ensure roles are loaded
        db.session.refresh(target_user)
        
        logger.info(f"Admin {user.id} assigned role '{role.name}' to user {target_user.id}")
        return jsonify({
            "message": f"Role '{role.name}' assigned successfully",
            "user": target_user.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error assigning role: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to assign role"}), 500


@app.delete("/api/roles/<int:role_id>/assign")
@admin_required
def remove_role(role_id: int):
    """Remove a role from a user - admin only."""
    try:
        user = current_user()
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        data = ensure_json_request()
        user_id = data.get("user_id")
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Get role and target user
        role = db.session.get(Role, role_id)
        if not role:
            return jsonify({"error": "Role not found"}), 404
        
        target_user = db.session.get(User, user_id)
        if not target_user:
            return jsonify({"error": "User not found"}), 404
        
        # Check if user has this role
        if role not in target_user.roles:
            return jsonify({"error": "User does not have this role"}), 400
        
        # Remove role
        target_user.roles.remove(role)
        db.session.commit()
        
        # Refresh the user object to ensure roles are updated
        db.session.refresh(target_user)
        
        logger.info(f"Admin {user.id} removed role '{role.name}' from user {target_user.id}")
        return jsonify({
            "message": f"Role '{role.name}' removed successfully",
            "user": target_user.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error removing role: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to remove role"}), 500


@app.post("/api/roles/ai-suggest")
@login_required
def ai_suggest_role_instructions():
    """Get AI suggestions for role instructions - anyone can use."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        data = ensure_json_request()
        role_name = data.get("role_name", "").strip()
        role_description = data.get("role_description", "").strip()
        
        if not role_name:
            return jsonify({"error": "Role name is required"}), 400
        
        # Create prompt for AI
        prompt = f"""Create detailed instructions for what should happen when a user is assigned the role "{role_name}"."""
        if role_description:
            prompt += f"\n\nRole description: {role_description}"
        prompt += "\n\nProvide clear, actionable instructions on:\n1. What permissions or access this role grants\n2. What features or sections become available\n3. What actions the user can perform\n4. Any restrictions or limitations\n\nKeep it concise but comprehensive."
        
        # Call AI to generate instructions
        ai_instructions = call_openai(
            [{"role": "user", "content": prompt}],
            system_prompt="You are an expert at creating role-based access control instructions. Provide clear, actionable guidance."
        )
        
        return jsonify({
            "ai_instructions": ai_instructions.strip()
        })
    except Exception as e:
        logger.exception(f"Error generating AI suggestions: {e}")
        return jsonify({"error": "Failed to generate AI suggestions"}), 500


###############################################################################
# Cloud PCs Management                                                         #
###############################################################################


@app.get("/api/cloud-pcs")
@login_required
def list_cloud_pcs():
    """Get all cloud PCs for the current user."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pcs = db.session.query(CloudPC).filter_by(owner_id=user.id).order_by(CloudPC.created_at.desc()).all()
        return jsonify({"cloud_pcs": [pc.to_dict() for pc in cloud_pcs]})
    except Exception as e:
        logger.exception(f"Error listing cloud PCs: {e}")
        return jsonify({"error": "Failed to load cloud PCs"}), 500


@app.post("/api/cloud-pcs")
@login_required
def create_cloud_pc():
    """Create a new cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        data = ensure_json_request()
        name = data.get("name", "").strip()
        os_version = data.get("os_version", "1.0 beta").strip()
        
        if not name:
            return jsonify({"error": "VM name is required"}), 400
        
        if os_version not in ["1.0 beta"]:
            return jsonify({"error": "Only '1.0 beta' OS is available"}), 400
        
        # Insert with explicit storage_gb for backward compatibility with old schema
        # Use raw SQL to handle old schema that requires storage_gb (NOT NULL)
        try:
            result = db.session.execute(
                db.text("""
                    INSERT INTO cloud_pcs (owner_id, name, os_version, status, storage_used_mb, storage_gb, created_at, updated_at)
                    VALUES (:owner_id, :name, :os_version, :status, :storage_used_mb, 0, datetime('now'), datetime('now'))
                """),
                {
                    "owner_id": user.id,
                    "name": name,
                    "os_version": os_version,
                    "status": "created",
                    "storage_used_mb": 0
                }
            )
            pc_id = result.lastrowid
            db.session.commit()
            
            # Fetch the created CloudPC
            cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id).first()
        except Exception as e:
            # Fallback to normal ORM insert if storage_gb doesn't exist or other error
            logger.debug(f"Fallback to ORM insert: {e}")
            cloud_pc = CloudPC(
                owner_id=user.id,
                name=name,
                os_version=os_version,
                status="created",
                storage_used_mb=0
            )
            db.session.add(cloud_pc)
            db.session.commit()
            db.session.refresh(cloud_pc)
        
        logger.info(f"User {user.id} created cloud PC '{name}' with OS {os_version}")
        
        return jsonify({
            "message": "Cloud PC created successfully",
            "cloud_pc": cloud_pc.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error creating cloud PC: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to create cloud PC"}), 500


@app.get("/api/cloud-pcs/<int:pc_id>")
@login_required
def get_cloud_pc(pc_id: int):
    """Get a specific cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        return jsonify({"cloud_pc": cloud_pc.to_dict()})
    except Exception as e:
        logger.exception(f"Error getting cloud PC: {e}")
        return jsonify({"error": "Failed to load cloud PC"}), 500


@app.post("/api/cloud-pcs/<int:pc_id>/verify-password")
@login_required
def verify_cloud_pc_password(pc_id: int):
    """Verify password to access cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        password = data.get("password", "")
        
        # Verify password against user's account password
        if not verify_password(password, user.password_hash):
            return jsonify({"error": "Incorrect password"}), 401
        
        cloud_pc.status = "running"
        db.session.commit()
        
        return jsonify({
            "message": "Password verified",
            "cloud_pc": cloud_pc.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error verifying password: {e}")
        return jsonify({"error": "Failed to verify password"}), 500


@app.get("/api/cloud-pcs/<int:pc_id>/files")
@login_required
def list_cloud_pc_files(pc_id: int):
    """List files in cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        path = request.args.get("path", "/")
        # All data is stored in VM-specific directory linked to account
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        full_path = os.path.join(storage_dir, path.lstrip("/"))
        
        # Security: ensure path is within VM's storage directory (linked to account)
        if not os.path.abspath(full_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        os.makedirs(full_path, exist_ok=True)
        
        files = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            files.append({
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0,
                "modified": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
            })
        
        return jsonify({"files": files, "path": path})
    except Exception as e:
        logger.exception(f"Error listing files: {e}")
        return jsonify({"error": "Failed to list files"}), 500


@app.post("/api/cloud-pcs/<int:pc_id>/files/upload")
@login_required
def upload_cloud_pc_file(pc_id: int):
    """Upload file to cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        path = request.form.get("path", "/")
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        full_path = os.path.join(storage_dir, path.lstrip("/"))
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        os.makedirs(full_path, exist_ok=True)
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        filename = file.filename
        file_path = os.path.join(full_path, filename)
        file.save(file_path)
        
        # Update storage used
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        cloud_pc.storage_used_mb += int(file_size_mb)
        db.session.commit()
        
        return jsonify({"message": "File uploaded successfully"})
    except Exception as e:
        logger.exception(f"Error uploading file: {e}")
        return jsonify({"error": "Failed to upload file"}), 500


@app.post("/api/cloud-pcs/<int:pc_id>/files/create")
@login_required
def create_cloud_pc_file(pc_id: int):
    """Create a new file or directory in cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        path = data.get("path", "/")
        name = data.get("name", "").strip()
        type = data.get("type", "file")  # file or directory
        content = data.get("content", "")
        
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        full_path = os.path.join(storage_dir, path.lstrip("/"))
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        os.makedirs(full_path, exist_ok=True)
        
        item_path = os.path.join(full_path, name)
        
        if type == "directory":
            os.makedirs(item_path, exist_ok=True)
        else:
            with open(item_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size_mb = os.path.getsize(item_path) / (1024 * 1024)
            cloud_pc.storage_used_mb += int(file_size_mb)
            db.session.commit()
        
        return jsonify({"message": f"{type.capitalize()} created successfully"})
    except Exception as e:
        logger.exception(f"Error creating file: {e}")
        return jsonify({"error": "Failed to create file"}), 500


@app.get("/api/cloud-pcs/<int:pc_id>/files/read")
@login_required
def read_cloud_pc_file(pc_id: int):
    """Read file content from cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        path = request.args.get("path", "")
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        file_path = os.path.join(storage_dir, path.lstrip("/"))
        
        if not os.path.abspath(file_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Check if it's a binary file (image, etc.)
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        is_binary = mime_type and mime_type.startswith(('image/', 'video/', 'audio/', 'application/'))
        
        if is_binary:
            # For binary files, return as base64
            import base64
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                content = base64.b64encode(file_bytes).decode('utf-8')
            return jsonify({
                "content": content,
                "is_binary": True,
                "mime_type": mime_type,
                "filename": os.path.basename(file_path)
            })
        else:
            # Read text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding or return as binary
                import base64
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                    content = base64.b64encode(file_bytes).decode('utf-8')
                return jsonify({
                    "content": content,
                    "is_binary": True,
                    "mime_type": "application/octet-stream",
                    "filename": os.path.basename(file_path)
                })
            
            return jsonify({"content": content, "is_binary": False})
    except Exception as e:
        logger.exception(f"Error reading file: {e}")
        return jsonify({"error": "Failed to read file"}), 500


@app.put("/api/cloud-pcs/<int:pc_id>/files")
@login_required
def update_cloud_pc_file(pc_id: int):
    """Update file content in cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        path = data.get("path", "")
        content = data.get("content", "")
        
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        file_path = os.path.join(storage_dir, path.lstrip("/"))
        
        if not os.path.abspath(file_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            return jsonify({"error": "File not found"}), 404
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({"message": "File updated successfully"})
    except Exception as e:
        logger.exception(f"Error updating file: {e}")
        return jsonify({"error": "Failed to update file"}), 500


@app.put("/api/cloud-pcs/<int:pc_id>/files/rename")
@login_required
def rename_cloud_pc_file(pc_id: int):
    """Rename a file or directory in cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        old_path = data.get("old_path", "").strip()
        new_path = data.get("new_path", "").strip()
        
        if not old_path or not new_path:
            return jsonify({"error": "Old path and new path are required"}), 400
        
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        old_file_path = os.path.join(storage_dir, old_path.lstrip("/"))
        new_file_path = os.path.join(storage_dir, new_path.lstrip("/"))
        
        # Security: ensure paths are within VM's storage directory
        if not os.path.abspath(old_file_path).startswith(os.path.abspath(storage_dir)) or \
           not os.path.abspath(new_file_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        if not os.path.exists(old_file_path):
            return jsonify({"error": "File or directory not found"}), 404
        
        if os.path.exists(new_file_path):
            return jsonify({"error": "A file or directory with that name already exists"}), 400
        
        # Rename file or directory
        os.rename(old_file_path, new_file_path)
        
        logger.info(f"User {user.id} renamed {old_path} to {new_path} in cloud PC {pc_id}")
        return jsonify({"message": "File renamed successfully"})
    except Exception as e:
        logger.exception(f"Error renaming file: {e}")
        return jsonify({"error": "Failed to rename file"}), 500


@app.delete("/api/cloud-pcs/<int:pc_id>/files")
@login_required
def delete_cloud_pc_file(pc_id: int):
    """Delete a file or directory in cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        path = data.get("path", "").strip()
        
        if not path:
            return jsonify({"error": "Path is required"}), 400
        
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        full_path = os.path.join(storage_dir, path.lstrip("/"))
        
        # Security: ensure path is within VM's storage directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        if not os.path.exists(full_path):
            return jsonify({"error": "File or directory not found"}), 404
        
        # Calculate size before deletion for storage update
        file_size_mb = 0
        if os.path.isfile(full_path):
            file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        elif os.path.isdir(full_path):
            import shutil
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    file_size_mb += os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
        
        # Delete file or directory
        if os.path.isfile(full_path):
            os.remove(full_path)
        else:
            import shutil
            shutil.rmtree(full_path)
        
        # Update storage used
        cloud_pc.storage_used_mb = max(0, cloud_pc.storage_used_mb - int(file_size_mb))
        db.session.commit()
        
        logger.info(f"User {user.id} deleted {path} from cloud PC {pc_id}")
        return jsonify({"message": "File deleted successfully"})
    except Exception as e:
        logger.exception(f"Error deleting file: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to delete file"}), 500


@app.post("/api/cloud-pcs/<int:pc_id>/files/drawing")
@login_required
def save_cloud_pc_drawing(pc_id: int):
    """Save drawing as PNG to cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        data = ensure_json_request()
        path = data.get("path", "/")
        filename = data.get("filename", "drawing.png")
        image_data = data.get("image_data", "")  # Base64 encoded image
        
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}", "storage")
        full_path = os.path.join(storage_dir, path.lstrip("/"))
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(storage_dir)):
            return jsonify({"error": "Invalid path"}), 400
        
        os.makedirs(full_path, exist_ok=True)
        
        # Decode base64 image
        import base64
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        file_path = os.path.join(full_path, filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Update storage used
        file_size_mb = len(image_bytes) / (1024 * 1024)
        cloud_pc.storage_used_mb += int(file_size_mb)
        db.session.commit()
        
        return jsonify({"message": "Drawing saved successfully"})
    except Exception as e:
        logger.exception(f"Error saving drawing: {e}")
        return jsonify({"error": "Failed to save drawing"}), 500


@app.delete("/api/cloud-pcs/<int:pc_id>")
@login_required
def delete_cloud_pc(pc_id: int):
    """Delete a cloud PC."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        cloud_pc = db.session.query(CloudPC).filter_by(id=pc_id, owner_id=user.id).first()
        if not cloud_pc:
            return jsonify({"error": "Cloud PC not found"}), 404
        
        # Delete storage directory
        storage_dir = os.path.join(UPLOAD_ROOT, "cloud_pcs", f"pc_{pc_id}")
        if os.path.exists(storage_dir):
            try:
                import shutil
                shutil.rmtree(storage_dir)
            except Exception as e:
                logger.warning(f"Failed to delete storage directory: {e}")
        
        db.session.delete(cloud_pc)
        db.session.commit()
        
        logger.info(f"User {user.id} deleted cloud PC {pc_id}")
        return jsonify({"message": "Cloud PC deleted successfully"})
    except Exception as e:
        logger.exception(f"Error deleting cloud PC: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to delete cloud PC"}), 500


###############################################################################
# AI App Creation                                                              #
###############################################################################


@app.post("/api/ai-apps")
@login_required
def create_ai_app():
    """Create a new AI app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        data = ensure_json_request()
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        
        if not name:
            return jsonify({"error": "App name is required"}), 400
        
        ai_app = AIApp(
            developer_id=user.id,
            name=name,
            description=description,
            code="",
            is_live=False
        )
        db.session.add(ai_app)
        db.session.commit()
        db.session.refresh(ai_app)
        
        logger.info(f"User {user.id} created AI app '{name}'")
        
        return jsonify({
            "message": "AI app created successfully",
            "app": ai_app.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error creating AI app: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to create AI app"}), 500


@app.get("/api/ai-apps")
@login_required
def list_ai_apps():
    """Get all AI apps - developer's apps or live apps."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get developer's own apps (including non-live)
        my_apps = db.session.query(AIApp).filter_by(developer_id=user.id).order_by(AIApp.created_at.desc()).all()
        
        # Get all live apps (visible to community)
        live_apps = db.session.query(AIApp).filter_by(is_live=True).order_by(AIApp.live_at.desc()).all()
        
        return jsonify({
            "my_apps": [app.to_dict() for app in my_apps],
            "live_apps": [app.to_dict() for app in live_apps]
        })
    except Exception as e:
        logger.exception(f"Error listing AI apps: {e}")
        return jsonify({"error": "Failed to load AI apps"}), 500


@app.get("/api/ai-apps/<int:app_id>")
@login_required
def get_ai_app(app_id: int):
    """Get a specific AI app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        # Only developer can see non-live apps
        if not ai_app.is_live and ai_app.developer_id != user.id:
            return jsonify({"error": "Access denied"}), 403
        
        return jsonify({"app": ai_app.to_dict()})
    except Exception as e:
        logger.exception(f"Error getting AI app: {e}")
        return jsonify({"error": "Failed to load AI app"}), 500


@app.put("/api/ai-apps/<int:app_id>")
@login_required
def update_ai_app(app_id: int):
    """Update an AI app (code, name, description)."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id, developer_id=user.id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        data = ensure_json_request()
        
        if "name" in data:
            ai_app.name = data["name"].strip()
        if "description" in data:
            ai_app.description = data.get("description", "").strip()
        if "code" in data:
            ai_app.code = data["code"]
        
        db.session.commit()
        
        logger.info(f"User {user.id} updated AI app {app_id}")
        
        return jsonify({
            "message": "AI app updated successfully",
            "app": ai_app.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error updating AI app: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to update AI app"}), 500


@app.post("/api/ai-apps/<int:app_id>/generate")
@login_required
def generate_ai_app_code(app_id: int):
    """Use AI to generate app code based on description."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id, developer_id=user.id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        data = ensure_json_request()
        prompt = data.get("prompt", "").strip()
        
        if not prompt:
            prompt = ai_app.description or f"Create a {ai_app.name} app"
        
        # Check if this is for Cloud PC
        pc_id = data.get("pc_id")
        cloud_pc_context = ""
        if pc_id:
            pc_id_str = str(pc_id)
            cloud_pc_context = f"""
IMPORTANT: This app is running inside a Cloud PC environment. You MUST use Cloud PC file APIs instead of regular file system APIs.

Cloud PC File API Endpoints:
- GET /api/cloud-pcs/{pc_id_str}/files?path=/path - List files and folders
- GET /api/cloud-pcs/{pc_id_str}/files/read?path=/path/to/file - Read file content (returns JSON with 'content' and 'is_binary' fields)
- POST /api/cloud-pcs/{pc_id_str}/files - Upload/create file (form-data with 'file' and 'path')
- POST /api/cloud-pcs/{pc_id_str}/files/drawing - Save drawing/image (JSON with 'path', 'filename', 'image_data' as base64)
- PUT /api/cloud-pcs/{pc_id_str}/files/rename - Rename file (JSON with 'old_path' and 'new_path')
- DELETE /api/cloud-pcs/{pc_id_str}/files?path=/path - Delete file or folder

File Reading Example:
```javascript
// window.CLOUD_PC_ID is automatically available in your app
const PC_ID = window.CLOUD_PC_ID; // Cloud PC ID is injected automatically
// For text files
const response = await fetch('/api/cloud-pcs/' + PC_ID + '/files/read?path=' + encodeURIComponent(filePath));
const data = await response.json();
const content = data.content; // Text content

// For binary files (images, etc.)
const response = await fetch('/api/cloud-pcs/' + PC_ID + '/files/read?path=' + encodeURIComponent(filePath));
const data = await response.json();
if (data.is_binary) {{
  const imageUrl = 'data:' + data.mime_type + ';base64,' + data.content;
  // Use imageUrl in img src
}}
```

File Listing Example:
```javascript
// window.CLOUD_PC_ID is automatically available in your app
const PC_ID = window.CLOUD_PC_ID; // Cloud PC ID is injected automatically
const response = await fetch('/api/cloud-pcs/' + PC_ID + '/files?path=' + encodeURIComponent(currentPath));
const data = await response.json();
const files = data.files; // Array of {{name, type: 'file'|'directory', size}}
```

DO NOT use:
- FileReader API for local files
- <input type="file"> for browsing local computer files
- window.showOpenFilePicker() or similar browser file APIs
- Any Node.js file system APIs (fs module)

INSTEAD use:
- Cloud PC file API endpoints shown above
- Fetch API to call Cloud PC endpoints
- The Cloud PC file browser APIs for file selection
"""

        # Use AI to generate app code
        system_prompt = f"""You are an expert web developer. Generate complete, working HTML/CSS/JavaScript code for web applications running in a Cloud PC environment.
        
Requirements:
- Return ONLY valid HTML code (can include <style> and <script> tags)
- Make it modern, beautiful, and functional
- Use modern CSS (flexbox, grid, gradients)
- Include interactive JavaScript functionality
- Make it responsive and user-friendly
- No external dependencies unless necessary
- Return complete, runnable code
{cloud_pc_context if pc_id else ''}

CRITICAL: If the app needs to browse, read, or manage files, you MUST use Cloud PC file APIs (shown above) instead of local file system APIs."""
        
        full_prompt = f"""Create a web application: {prompt}

App Name: {ai_app.name}
Description: {ai_app.description or 'No description provided'}
{cloud_pc_context if pc_id else ''}

Generate complete HTML code with embedded CSS and JavaScript. Make it modern, beautiful, and fully functional."""
        
        generated_code = call_openai(
            [{"role": "user", "content": full_prompt}],
            system_prompt=system_prompt
        )
        
        # Inject Cloud PC ID into the code if it's for Cloud PC
        if pc_id:
            # Inject a script tag at the beginning to provide Cloud PC ID
            pc_id_injection = f"""<script>
// Cloud PC Context - This app runs in Cloud PC ID: {pc_id}
window.CLOUD_PC_ID = {pc_id};
</script>
"""
            # Insert after <html> tag or at the beginning if no html tag
            if "<html" in generated_code.lower():
                # Find the opening html tag and insert after it
                html_tag_match = re.search(r"<html[^>]*>", generated_code, re.IGNORECASE)
                if html_tag_match:
                    insert_pos = html_tag_match.end()
                    generated_code = generated_code[:insert_pos] + "\n" + pc_id_injection + generated_code[insert_pos:]
                else:
                    generated_code = pc_id_injection + generated_code
            else:
                generated_code = pc_id_injection + generated_code
        
        # Update app code
        ai_app.code = generated_code
        db.session.commit()
        
        logger.info(f"User {user.id} generated code for AI app {app_id}")
        
        return jsonify({
            "message": "Code generated successfully",
            "code": generated_code,
            "app": ai_app.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error generating AI app code: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to generate code"}), 500


@app.post("/api/ai-apps/<int:app_id>/go-live")
@login_required
def make_ai_app_live(app_id: int):
    """Make an AI app live (visible to community)."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id, developer_id=user.id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        if not ai_app.code:
            return jsonify({"error": "Cannot go live: App code is empty"}), 400
        
        ai_app.is_live = True
        ai_app.live_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"User {user.id} made AI app {app_id} live")
        
        return jsonify({
            "message": "App is now live!",
            "app": ai_app.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error making AI app live: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to make app live"}), 500


@app.delete("/api/ai-apps/<int:app_id>")
@login_required
def delete_ai_app(app_id: int):
    """Delete an AI app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id, developer_id=user.id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        # Delete all chat messages for this app
        db.session.query(AIAppChat).filter_by(app_id=app_id).delete()
        
        db.session.delete(ai_app)
        db.session.commit()
        
        logger.info(f"User {user.id} deleted AI app {app_id}")
        
        return jsonify({"message": "AI app deleted successfully"})
    except Exception as e:
        logger.exception(f"Error deleting AI app: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to delete AI app"}), 500


@app.get("/api/ai-apps/<int:app_id>/chat")
@login_required
def get_ai_app_chat(app_id: int):
    """Get chat messages for an AI app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        # Only developer can see chat for non-live apps
        if not ai_app.is_live and ai_app.developer_id != user.id:
            return jsonify({"error": "Access denied"}), 403
        
        # Get chat messages for this app and user
        chats = db.session.query(AIAppChat).filter_by(
            app_id=app_id,
            user_id=user.id
        ).order_by(AIAppChat.created_at.asc()).all()
        
        return jsonify({
            "messages": [chat.to_dict() for chat in chats]
        })
    except Exception as e:
        logger.exception(f"Error getting chat: {e}")
        return jsonify({"error": "Failed to load chat"}), 500


@app.post("/api/ai-apps/<int:app_id>/chat")
@login_required
def send_ai_app_chat_message(app_id: int):
    """Send a chat message to AI about the app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        ai_app = db.session.query(AIApp).filter_by(id=app_id).first()
        if not ai_app:
            return jsonify({"error": "AI app not found"}), 404
        
        # Only developer can chat about non-live apps
        if not ai_app.is_live and ai_app.developer_id != user.id:
            return jsonify({"error": "Access denied"}), 403
        
        data = ensure_json_request()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Save user message
        user_chat = AIAppChat(
            app_id=app_id,
            user_id=user.id,
            message=message,
            is_user_message=True
        )
        db.session.add(user_chat)
        db.session.flush()
        
        # Get chat history for context
        previous_chats = db.session.query(AIAppChat).filter_by(
            app_id=app_id,
            user_id=user.id
        ).order_by(AIAppChat.created_at.desc()).limit(10).all()
        
        # Build context for AI
        chat_history = []
        for chat in reversed(previous_chats):
            if chat.is_user_message:
                chat_history.append({"role": "user", "content": chat.message})
            elif chat.response:
                chat_history.append({"role": "assistant", "content": chat.response})
        
        # Add current message
        chat_history.append({"role": "user", "content": message})
        
        # Check if this is for Cloud PC
        pc_id = data.get("pc_id")
        cloud_pc_context = ""
        if pc_id:
            pc_id_str = str(pc_id)
            cloud_pc_context = f"""

IMPORTANT CONTEXT: This app is running inside a Cloud PC environment (Cloud PC ID: {pc_id_str}).

The app MUST use Cloud PC file APIs instead of regular file system APIs:
- GET /api/cloud-pcs/{pc_id_str}/files?path=/path - List files
- GET /api/cloud-pcs/{pc_id_str}/files/read?path=/path - Read file (returns JSON with 'content' and 'is_binary')
- POST /api/cloud-pcs/{pc_id_str}/files - Upload/create file
- PUT /api/cloud-pcs/{pc_id_str}/files/rename - Rename file
- DELETE /api/cloud-pcs/{pc_id_str}/files?path=/path - Delete file

DO NOT use: FileReader API, <input type="file"> for local files, window.showOpenFilePicker(), or any Node.js fs APIs.
USE INSTEAD: Fetch API calls to Cloud PC endpoints shown above.

When fixing file-related code, ensure it uses Cloud PC APIs, not local file system APIs.
"""

        # System prompt for AI app assistant
        system_prompt = f"""You are an AI app builder that AUTOMATICALLY builds complete web applications. You are NOT helping someone code manually - YOU are doing all the coding yourself.

IMPORTANT: The user describes what they want, and YOU build the entire app automatically. They do NOT write code - YOU do.

App Name: {ai_app.name}
App Description: {ai_app.description or 'No description provided'}
{cloud_pc_context}

Current App Code:
```html
{ai_app.code[:2000] if ai_app.code else 'No code yet'}
```

CRITICAL INSTRUCTIONS - YOU ARE THE BUILDER:
- When the user asks for ANYTHING (build, create, fix, modify, add feature), YOU must write the COMPLETE working code
- DO NOT tell them what to code or how to code it - YOU code it for them
- DO NOT say "you need to add..." or "you should create..." - Instead, YOU add it and provide the complete code
- The code you provide will be automatically applied - so provide complete, runnable HTML/CSS/JavaScript
- Include ALL necessary code (HTML structure, CSS styles, JavaScript functionality) in a single code block
- If fixing bugs or making changes, provide the ENTIRE updated code, not instructions
- When asked to build something new, provide the COMPLETE app code from scratch
- Ensure file operations use Cloud PC APIs, not local file system APIs

Your role as the AI Builder:
- YOU build complete, working applications automatically
- YOU fix bugs by writing the complete corrected code
- YOU add features by writing the complete updated code
- Answer questions about the app when asked
- Always provide working code, never just instructions

EXAMPLE INTERACTIONS:
User: "Build a todo app"
You: [Provide complete HTML/CSS/JS code in ```html block]

User: "Add a dark mode button"
You: [Provide complete updated code with dark mode feature in ```html block]

User: "Fix the bug where items don't delete"
You: [Provide complete fixed code in ```html block]

ALWAYS wrap your code in ```html code blocks like this:
```html
<!DOCTYPE html>
<html>
<head>
  <style>
    /* CSS here */
  </style>
</head>
<body>
  <!-- HTML here -->
  <script>
    // JavaScript here
  </script>
</body>
</html>
```

Remember: YOU are the developer. The user describes, YOU build."""
        
        # Get AI response
        try:
            ai_response = call_openai(chat_history, system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"OpenAI error in chat: {e}")
            ai_response = "I'm having trouble connecting to the AI service. Please try again."
        
        # Save AI response
        ai_chat = AIAppChat(
            app_id=app_id,
            user_id=user.id,
            message=message,
            response=ai_response,
            is_user_message=False
        )
        db.session.add(ai_chat)
        db.session.commit()
        
        logger.info(f"User {user.id} sent chat message for app {app_id}")
        
        return jsonify({
            "message": "Message sent successfully",
            "user_message": user_chat.to_dict(),
            "ai_response": ai_chat.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error sending chat message: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to send message"}), 500


###############################################################################
# App Store                                                                    #
###############################################################################


@app.get("/api/app-store")
@login_required
def get_app_store():
    """Get all live apps for the app store."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get all live apps
        live_apps = db.session.query(AIApp).filter_by(is_live=True).order_by(AIApp.live_at.desc()).all()
        
        # Get user's downloaded app IDs
        downloaded_app_ids = {
            download.app_id 
            for download in db.session.query(AIAppDownload).filter_by(user_id=user.id).all()
        }
        
        # Get developer info for each app
        apps_with_developer = []
        for app in live_apps:
            developer = db.session.query(User).filter_by(id=app.developer_id).first()
            app_dict = app.to_dict()
            app_dict["developer"] = developer.username if developer else "Unknown"
            app_dict["is_downloaded"] = app.id in downloaded_app_ids
            apps_with_developer.append(app_dict)
        
        return jsonify({
            "apps": apps_with_developer
        })
    except Exception as e:
        logger.exception(f"Error getting app store: {e}")
        return jsonify({"error": "Failed to load app store"}), 500


@app.post("/api/app-store/<int:app_id>/download")
@login_required
def download_app(app_id: int):
    """Download/install an app from the app store."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Check if app exists and is live
        ai_app = db.session.query(AIApp).filter_by(id=app_id, is_live=True).first()
        if not ai_app:
            return jsonify({"error": "App not found or not available"}), 404
        
        # Check if already downloaded
        existing_download = db.session.query(AIAppDownload).filter_by(
            user_id=user.id,
            app_id=app_id
        ).first()
        
        if existing_download:
            return jsonify({"error": "App already downloaded"}), 400
        
        # Create download record
        download = AIAppDownload(
            user_id=user.id,
            app_id=app_id
        )
        db.session.add(download)
        db.session.commit()
        
        logger.info(f"User {user.id} downloaded app {app_id}")
        
        return jsonify({
            "message": "App downloaded successfully",
            "download": download.to_dict()
        })
    except Exception as e:
        logger.exception(f"Error downloading app: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to download app"}), 500


@app.get("/api/app-store/my-downloads")
@login_required
def get_my_downloads():
    """Get user's downloaded apps."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        # Get user's downloads
        downloads = db.session.query(AIAppDownload).filter_by(user_id=user.id).order_by(AIAppDownload.created_at.desc()).all()
        
        # Get app details for each download
        downloaded_apps = []
        for download in downloads:
            app = db.session.query(AIApp).filter_by(id=download.app_id).first()
            if app:
                developer = db.session.query(User).filter_by(id=app.developer_id).first()
                app_dict = app.to_dict()
                app_dict["developer"] = developer.username if developer else "Unknown"
                app_dict["downloaded_at"] = download.created_at.isoformat() if download.created_at else None
                downloaded_apps.append(app_dict)
        
        return jsonify({
            "apps": downloaded_apps
        })
    except Exception as e:
        logger.exception(f"Error getting downloads: {e}")
        return jsonify({"error": "Failed to load downloads"}), 500


@app.delete("/api/app-store/<int:app_id>/download")
@login_required
def remove_download(app_id: int):
    """Remove a downloaded app."""
    try:
        user = current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        download = db.session.query(AIAppDownload).filter_by(
            user_id=user.id,
            app_id=app_id
        ).first()
        
        if not download:
            return jsonify({"error": "App not found in downloads"}), 404
        
        db.session.delete(download)
        db.session.commit()
        
        logger.info(f"User {user.id} removed download for app {app_id}")
        
        return jsonify({"message": "App removed from downloads"})
    except Exception as e:
        logger.exception(f"Error removing download: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to remove download"}), 500


###############################################################################
# Error handling                                                               #
###############################################################################


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file size limit errors."""
    max_size_gb = app.config.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024 * 1024) / (1024 * 1024 * 1024)
    return jsonify({
        "error": f"File is too large. Maximum file size is {max_size_gb:.1f} GB. Please compress your video or use a smaller file."
    }), 413


@app.errorhandler(Exception)
def handle_exception(e):  # pragma: no cover - global safeguard
    """Global error handler that always returns a response."""
    # Don't handle HTTP exceptions (they're already proper responses)
    from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
    if isinstance(e, RequestEntityTooLarge):
        max_size_gb = app.config.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024 * 1024) / (1024 * 1024 * 1024)
        try:
            return jsonify({
                "error": f"File is too large. Maximum file size is {max_size_gb:.1f} GB. Please compress your video or use a smaller file."
            }), 413
        except:
            return make_response(f"File too large. Max size: {max_size_gb:.1f} GB", 413)
    if isinstance(e, HTTPException):
        try:
            response = jsonify({"error": e.description})
            response.status_code = e.code
            return response
        except Exception as http_error:
            logger.exception(f"Error handling HTTPException: {http_error}")
            try:
                return make_response(json.dumps({"error": "An error occurred"}), e.code, {"Content-Type": "application/json"})
            except:
                return make_response("An error occurred", e.code)
    
    # Log the exception
    try:
        logger.exception("Unhandled exception: %s", e)
    except:
        pass  # If logging fails, continue anyway
    
    # Try to get a user-friendly error message
    user_message = "Something went wrong. Please try again or contact support."
    try:
        user_message = transform_error_for_user(str(e))
    except Exception as transform_error:
        # If error transformation fails, use a safe fallback
        try:
            logger.error(f"Error transformation failed: {transform_error}")
        except:
            pass
    
    # Always return a response, even if jsonify fails
    try:
        response = jsonify({"error": user_message})
        response.status_code = 500
        return response
    except Exception as response_error:
        # If even creating the response fails, return a minimal response
        try:
            logger.exception(f"Error creating error response: {response_error}")
        except:
            pass
        try:
            return make_response(json.dumps({"error": user_message}), 500, {"Content-Type": "application/json"})
        except:
            try:
                return make_response("Internal Server Error", 500)
            except:
                # Absolute last resort - return a simple string
                return "Internal Server Error", 500


@app.after_request
def apply_cors_headers(response):
    """Apply CORS headers to all responses (only if not already set by flask-cors)."""
    try:
        origin = request.headers.get("Origin")
        if origin and cors_origin_check(origin):
            # Only set CORS headers if flask-cors hasn't already set them
            if "Access-Control-Allow-Origin" not in response.headers:
                response.headers.add("Access-Control-Allow-Origin", origin)
            if "Access-Control-Allow-Credentials" not in response.headers:
                response.headers.add("Access-Control-Allow-Credentials", "true")
            if "Access-Control-Allow-Headers" not in response.headers:
                response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, Range")
            if "Access-Control-Allow-Methods" not in response.headers:
                response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD")
        return response
    except Exception as e:
        logger.error(f"Error applying CORS headers: {e}")
        return response


###############################################################################
# CLI / bootstrap                                                              #
###############################################################################


@app.cli.command("init-db")
def init_db_command():
    """Initialize the database tables."""
    try:
        logger.info(f"Initializing database at: {DATABASE_PATH}")
        db.create_all()
        
        # Ensure admin user exists
        admin = db.session.query(User).filter_by(username='admin').first()
        if not admin:
            logger.info("Creating default admin user...")
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=hash_password('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("✓ Default admin user created (username: admin, password: admin123)")
        else:
            print(f"✓ Admin user already exists (ID: {admin.id})")
        
        print(f"✓ Database initialized at: {DATABASE_PATH}")
    except Exception as e:
        logger.exception(f"Error initializing database: {e}")
        print(f"ERROR: Failed to initialize database: {e}")
        raise


if __name__ == "__main__":
    with app.app_context():
        try:
            logger.info(f"Initializing database at: {DATABASE_PATH}")
            db.create_all()
            logger.info("Database tables created/verified")
            
            # Migration: Ensure open_apps column exists in cloud_pcs table
            try:
                from sqlalchemy import inspect, text
                inspector = inspect(db.engine)
                if 'cloud_pcs' in inspector.get_table_names():
                    columns = [col['name'] for col in inspector.get_columns('cloud_pcs')]
                    if 'open_apps' not in columns:
                        db.session.execute(text('ALTER TABLE cloud_pcs ADD COLUMN open_apps TEXT'))
                        db.session.commit()
                        logger.info("Added open_apps column to cloud_pcs table")
            except Exception as e:
                logger.warning(f"Could not check/add open_apps column: {e}")
            
            # Ensure admin user exists
            admin = db.session.query(User).filter_by(username='admin').first()
            if not admin:
                logger.info("Creating default admin user...")
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=hash_password('admin123'),
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Default admin user created (username: admin, password: admin123)")
        except Exception as e:
            logger.exception(f"Error initializing database: {e}")
            print(f"ERROR: Failed to initialize database: {e}")
    
    port = int(os.environ.get("PORT", 5002))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
