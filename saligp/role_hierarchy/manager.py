"""
Phase 6: Role Hierarchy Module
Manages dynamic ownership using SQLite
"""
import logging
import sqlite3
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from config.config import ROLE_HIERARCHY_CONFIG

logger = logging.getLogger(__name__)


class RoleHierarchyManager:
    """
    Manages dynamic ownership and role hierarchy
    """

    def __init__(self):
        self.db_path = ROLE_HIERARCHY_CONFIG["db_path"]
        self.roles = ROLE_HIERARCHY_CONFIG["roles"]
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database"""
        logger.info("=" * 60)
        logger.info("PHASE 6: ROLE HIERARCHY OWNERSHIP MANAGEMENT")
        logger.info("=" * 60)

        logger.info(f"\n[1] Initializing database at: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Files table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id INTEGER NOT NULL,
                    sha256_hash TEXT UNIQUE NOT NULL,
                    filename TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Ownership table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ownership (
                    ownership_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_duplicate INTEGER DEFAULT 0,
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )

            # Duplicate registrations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS duplicate_registrations (
                    reg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id_original INTEGER NOT NULL,
                    pair_id_duplicate INTEGER NOT NULL,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

        logger.info("    Database initialized successfully")

    def register_user(self, user_id: str, role: str) -> bool:
        """Register a new user"""
        if role not in self.roles:
            logger.error(f"    Invalid role: {role}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (user_id, role) VALUES (?, ?)",
                    (user_id, role),
                )
                conn.commit()
            logger.info(f"    Registered user: {user_id} with role: {role}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"    User already exists: {user_id}")
            return False

    def assign_ownership(
        self, pair_id: int, user_id: str, is_duplicate: int = 0
    ) -> bool:
        """Assign file ownership to user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get user role
                cursor.execute(
                    "SELECT role FROM users WHERE user_id = ?", (user_id,)
                )
                result = cursor.fetchone()

                if not result:
                    logger.error(f"    User not found: {user_id}")
                    return False

                role = result[0]

                cursor.execute(
                    """
                    INSERT INTO ownership (pair_id, user_id, role, is_duplicate)
                    VALUES (?, ?, ?, ?)
                    """,
                    (pair_id, user_id, role, is_duplicate),
                )
                conn.commit()

            logger.info(
                f"    Assigned pair {pair_id} to user {user_id} "
                f"(duplicate: {is_duplicate})"
            )
            return True
        except Exception as e:
            logger.error(f"    Error assigning ownership: {e}")
            return False

    def register_duplicate(
        self, original_pair_id: int, duplicate_pair_id: int
    ) -> bool:
        """Register duplicate relationship"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO duplicate_registrations 
                    (pair_id_original, pair_id_duplicate)
                    VALUES (?, ?)
                    """,
                    (original_pair_id, duplicate_pair_id),
                )
                conn.commit()

            logger.info(
                f"    Registered duplicate: {original_pair_id} -> {duplicate_pair_id}"
            )
            return True
        except Exception as e:
            logger.error(f"    Error registering duplicate: {e}")
            return False

    def get_ownership(self, pair_id: int) -> List[Dict]:
        """Get ownership information for a pair"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id, role, is_duplicate, assigned_at
                    FROM ownership
                    WHERE pair_id = ?
                    """,
                    (pair_id,),
                )
                rows = cursor.fetchall()

            result = [
                {
                    "user_id": row[0],
                    "role": row[1],
                    "is_duplicate": row[2],
                    "assigned_at": row[3],
                }
                for row in rows
            ]

            return result
        except Exception as e:
            logger.error(f"    Error getting ownership: {e}")
            return []

    def transfer_ownership(
        self, pair_id: int, from_user: str, to_user: str
    ) -> bool:
        """Transfer ownership from one user to another"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get to_user role
                cursor.execute(
                    "SELECT role FROM users WHERE user_id = ?", (to_user,)
                )
                result = cursor.fetchone()

                if not result:
                    logger.error(f"    User not found: {to_user}")
                    return False

                new_role = result[0]

                # Update ownership
                cursor.execute(
                    """
                    UPDATE ownership
                    SET user_id = ?, role = ?, assigned_at = CURRENT_TIMESTAMP
                    WHERE pair_id = ? AND user_id = ?
                    """,
                    (to_user, new_role, pair_id, from_user),
                )
                conn.commit()

            logger.info(f"    Transferred pair {pair_id}: {from_user} -> {to_user}")
            return True
        except Exception as e:
            logger.error(f"    Error transferring ownership: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get ownership statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total users
                cursor.execute("SELECT COUNT(*) FROM users")
                total_users = cursor.fetchone()[0]

                # Total owned files
                cursor.execute("SELECT COUNT(*) FROM ownership")
                total_ownerships = cursor.fetchone()[0]

                # Duplicate count
                cursor.execute(
                    "SELECT COUNT(*) FROM ownership WHERE is_duplicate = 1"
                )
                duplicate_count = cursor.fetchone()[0]

                # Users by role
                cursor.execute(
                    "SELECT role, COUNT(*) FROM users GROUP BY role"
                )
                users_by_role = dict(cursor.fetchall())

            stats = {
                "total_users": total_users,
                "total_ownerships": total_ownerships,
                "duplicate_count": duplicate_count,
                "users_by_role": users_by_role,
            }

            return stats
        except Exception as e:
            logger.error(f"    Error getting statistics: {e}")
            return {}

    def print_statistics(self) -> None:
        """Print ownership statistics"""
        stats = self.get_statistics()

        logger.info("\n[STATISTICS]")
        logger.info(f"    Total Users: {stats.get('total_users', 0)}")
        logger.info(f"    Total Ownerships: {stats.get('total_ownerships', 0)}")
        logger.info(
            f"    Duplicate Registrations: {stats.get('duplicate_count', 0)}"
        )

        users_by_role = stats.get("users_by_role", {})
        for role, count in users_by_role.items():
            logger.info(f"      {role}: {count}")
