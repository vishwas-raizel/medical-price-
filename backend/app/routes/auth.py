import secrets
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.user import User, UserProfile, PasswordResetToken
from backend.app.schemas.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    MessageResponse
)
from backend.app.services.auth_service import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_user_optional
)
from backend.app.services.audit_service import log_audit_event
from backend.app.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, request: Request, db: Session = Depends(get_db)):
    clean_email = user_in.email.lower().strip()
    
    # 1. Enforce password complexity
    if len(user_in.password.strip()) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters in length."
        )

    # 2. Prevent duplicate email registrations
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists. Please sign in instead."
        )

    # 3. First user gets admin, subsequent get user role
    user_count = db.query(User).count()
    role = "admin" if user_count == 0 else (user_in.role or "user")

    # 4. Hash password with bcrypt
    hashed = get_password_hash(user_in.password.strip())
    new_user = User(
        email=clean_email,
        hashed_password=hashed,
        full_name=user_in.full_name.strip() if user_in.full_name else None,
        role=role,
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # 5. Create associated profile
    profile = UserProfile(user_id=new_user.id)
    db.add(profile)
    db.commit()

    # 6. Issue JWT Token immediately for auto-login
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": new_user.email, "role": new_user.role, "user_id": new_user.id},
        expires_delta=access_token_expires
    )

    log_audit_event(
        db,
        action="REGISTER",
        resource="users",
        user_id=new_user.id,
        ip_address=request.client.host if request.client else None,
        details={"email": new_user.email, "role": new_user.role}
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": new_user
    }

@router.post("/login", response_model=Token)
def login(login_data: UserLogin, request: Request, db: Session = Depends(get_db)):
    clean_email = login_data.email.lower().strip()
    
    # 1. Lookup user (case-insensitive)
    user = db.query(User).filter(User.email == clean_email).first()
    
    # 2. Validate password
    if not user or not verify_password(login_data.password.strip(), user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email credentials or incorrect password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 3. Check active state
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This user account has been deactivated. Please contact an administrator."
        )

    # 4. Issue JWT access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user.email, "role": user.role, "user_id": user.id},
        expires_delta=access_token_expires
    )

    log_audit_event(
        db,
        action="LOGIN",
        resource="auth",
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        details={"email": user.email}
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    new_password: Optional[str] = None

@router.put("/profile", response_model=UserResponse)
def update_profile(
    profile_data: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if profile_data.full_name is not None:
        current_user.full_name = profile_data.full_name.strip()
    if profile_data.new_password:
        if len(profile_data.new_password.strip()) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")
        current_user.hashed_password = get_password_hash(profile_data.new_password.strip())
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(req_data: ForgotPasswordRequest, request: Request, db: Session = Depends(get_db)):
    clean_email = req_data.email.lower().strip()
    user = db.query(User).filter(User.email == clean_email).first()

    reset_token_str = None
    if user and user.is_active:
        # Invalidate any previously unexpired tokens for this user
        db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.is_used == False
        ).update({"is_used": True})
        
        # Generate new cryptographically secure token
        reset_token_str = secrets.token_urlsafe(32)
        token_entry = PasswordResetToken(
            user_id=user.id,
            token=reset_token_str,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            is_used=False
        )
        db.add(token_entry)
        db.commit()

        log_audit_event(
            db,
            action="FORGOT_PASSWORD_REQUEST",
            resource="auth",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            details={"email": user.email}
        )

    # Security requirement: Never reveal whether an email is registered
    return {
        "message": "If this email is registered in our platform, a secure password reset link has been dispatched.",
        "reset_token": reset_token_str  # Accessible for local development / testing
    }

@router.post("/reset-password", response_model=MessageResponse)
def reset_password(req_data: ResetPasswordRequest, request: Request, db: Session = Depends(get_db)):
    if len(req_data.new_password.strip()) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters in length."
        )

    # Find valid unexpired token
    token_record = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == req_data.token,
        PasswordResetToken.is_used == False,
        PasswordResetToken.expires_at > datetime.utcnow()
    ).first()

    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The password reset token is invalid, expired, or has already been utilized."
        )

    user = db.query(User).filter(User.id == token_record.user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="User account not found.")

    # Update password
    user.hashed_password = get_password_hash(req_data.new_password.strip())
    token_record.is_used = True
    db.commit()

    log_audit_event(
        db,
        action="PASSWORD_RESET_SUCCESS",
        resource="auth",
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        details={"email": user.email}
    )

    return {
        "message": "Password has been successfully updated. You may now sign in with your new credentials."
    }

@router.post("/logout")
def logout(request: Request, current_user: Optional[User] = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    if current_user:
        log_audit_event(
            db,
            action="LOGOUT",
            resource="auth",
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            details={"email": current_user.email}
        )
    return {"message": "Session successfully terminated."}
