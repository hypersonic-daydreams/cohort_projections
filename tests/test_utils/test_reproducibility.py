"""
Unit tests for the reproducibility module.

Tests execution logging and database interaction with mocked database.
"""

import subprocess
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cohort_projections.utils import reproducibility


class TestGetDbConnection:
    """Tests for get_db_connection function."""

    @patch.object(reproducibility, "psycopg2")
    def test_successful_connection(self, mock_psycopg2: MagicMock) -> None:
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        result = reproducibility.get_db_connection()

        mock_psycopg2.connect.assert_called_once_with(dbname="cohort_projections_meta")
        assert result is mock_conn

    @patch.object(reproducibility.psycopg2, "connect")
    def test_connection_failure_returns_none(self, mock_connect: MagicMock) -> None:
        """Test that connection failure returns None."""
        import psycopg2

        mock_connect.side_effect = psycopg2.OperationalError("Connection refused")

        result = reproducibility.get_db_connection()

        assert result is None


class TestGetGitCommit:
    """Tests for get_git_commit function."""

    @patch.object(reproducibility.subprocess, "check_output")
    def test_returns_commit_hash(self, mock_check_output: MagicMock) -> None:
        """Test that git commit hash is returned."""
        expected_hash = "abc123def456789"
        mock_check_output.return_value = f"{expected_hash}\n".encode()

        result = reproducibility.get_git_commit()

        assert result == expected_hash

    @patch.object(reproducibility.subprocess, "check_output")
    def test_returns_unknown_on_error(self, mock_check_output: MagicMock) -> None:
        """Test that 'unknown' is returned when git command fails."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        result = reproducibility.get_git_commit()

        assert result == "unknown"

    @patch.object(reproducibility.subprocess, "check_output")
    def test_strips_whitespace(self, mock_check_output: MagicMock) -> None:
        """Test that commit hash whitespace is stripped."""
        mock_check_output.return_value = b"  abc123  \n"

        result = reproducibility.get_git_commit()

        assert result == "abc123"


class TestGetScriptId:
    """Tests for get_script_id function."""

    def test_returns_id_when_found(self) -> None:
        """Test that script ID is returned when found in database."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)

        # Use a path that would be relative to PROJECT_ROOT
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test_script.py"

        result = reproducibility.get_script_id(mock_cursor, script_path)

        assert result == 42
        mock_cursor.execute.assert_called_once()

    def test_returns_none_when_not_found(self) -> None:
        """Test that None is returned when script not in database."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        script_path = reproducibility.PROJECT_ROOT / "scripts" / "unknown_script.py"

        result = reproducibility.get_script_id(mock_cursor, script_path)

        assert result is None


class TestLogExecution:
    """Tests for log_execution context manager."""

    @pytest.fixture
    def mock_db_setup(self) -> Generator[dict[str, MagicMock], None, None]:
        """Set up mocked database connection and cursor."""
        with (
            patch.object(reproducibility, "get_db_connection") as mock_get_conn,
            patch.object(reproducibility, "get_git_commit") as mock_git,
            patch.object(reproducibility, "get_script_id") as mock_script_id,
        ):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            mock_get_conn.return_value = mock_conn
            mock_git.return_value = "abc123"
            mock_script_id.return_value = 1

            yield {
                "connection": mock_conn,
                "cursor": mock_cursor,
                "get_conn": mock_get_conn,
                "git": mock_git,
                "script_id": mock_script_id,
            }

    def test_successful_execution_logs_success(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that successful execution is logged with 'success' status."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

        with reproducibility.log_execution(script_path) as run_id:
            assert run_id is not None
            # Simulate successful execution

        # Should have called execute twice (insert and update)
        assert mock_db_setup["cursor"].execute.call_count == 2

        # Check that the update was for success status
        update_call = mock_db_setup["cursor"].execute.call_args_list[1]
        assert "success" in str(update_call)

        # Connection should be closed
        mock_db_setup["connection"].close.assert_called_once()

    def test_failed_execution_logs_failure(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that failed execution is logged with 'failed' status."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

        with (
            pytest.raises(ValueError, match="Test error"),
            reproducibility.log_execution(script_path) as run_id,
        ):
            assert run_id is not None
            raise ValueError("Test error")

        # Should have called execute twice (insert and update for failure)
        assert mock_db_setup["cursor"].execute.call_count >= 2

        # Check that the update was for failed status
        update_call = mock_db_setup["cursor"].execute.call_args_list[1]
        assert "failed" in str(update_call)

    def test_no_db_connection_yields_none(self) -> None:
        """Test that log_execution yields None when no DB connection."""
        with patch.object(reproducibility, "get_db_connection", return_value=None):
            script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

            with reproducibility.log_execution(script_path) as run_id:
                assert run_id is None

    def test_logs_parameters(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that parameters are logged correctly."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"
        params = {"year": 2024, "scenario": "baseline"}

        with reproducibility.log_execution(script_path, parameters=params):
            pass

        # Check that the insert call included the parameters
        insert_call = mock_db_setup["cursor"].execute.call_args_list[0]
        call_args = insert_call[0][1]  # The tuple of values
        assert '"year": 2024' in str(call_args)
        assert '"scenario": "baseline"' in str(call_args)

    def test_logs_inputs_and_outputs(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that input and output manifests are logged."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"
        inputs = [Path("/data/input1.csv"), Path("/data/input2.csv")]
        outputs = [Path("/data/output.csv")]

        with reproducibility.log_execution(script_path, inputs=inputs, outputs=outputs):
            pass

        # Check that the insert call included the manifests
        insert_call = mock_db_setup["cursor"].execute.call_args_list[0]
        call_args = insert_call[0][1]
        assert "input1.csv" in str(call_args)
        assert "input2.csv" in str(call_args)
        assert "output.csv" in str(call_args)

    def test_commits_on_success(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that database commits on successful execution."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

        with reproducibility.log_execution(script_path):
            pass

        # Should have committed twice (after insert and after update)
        assert mock_db_setup["connection"].commit.call_count == 2

    def test_generates_unique_run_id(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test that unique run IDs are generated."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"
        run_ids = []

        for _ in range(3):
            with reproducibility.log_execution(script_path) as run_id:
                run_ids.append(run_id)

        # All run IDs should be unique
        assert len(set(run_ids)) == 3

    def test_handles_empty_parameters(self, mock_db_setup: dict[str, MagicMock]) -> None:
        """Test handling of None/empty parameters."""
        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

        with reproducibility.log_execution(script_path, parameters=None):
            pass

        # Should not raise and should log empty dict
        insert_call = mock_db_setup["cursor"].execute.call_args_list[0]
        call_args = insert_call[0][1]
        assert "{}" in str(call_args)


class TestLogExecutionEdgeCases:
    """Edge case tests for log_execution."""

    @patch.object(reproducibility, "get_db_connection")
    @patch.object(reproducibility, "get_git_commit")
    @patch.object(reproducibility, "get_script_id")
    def test_db_error_during_update_does_not_raise(
        self,
        mock_script_id: MagicMock,
        mock_git: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Test that DB error during failure update doesn't mask original error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        mock_git.return_value = "abc123"
        mock_script_id.return_value = 1

        # First execute succeeds (insert), second fails (update on error)
        mock_cursor.execute.side_effect = [None, None, Exception("DB error")]

        script_path = reproducibility.PROJECT_ROOT / "scripts" / "test.py"

        with (
            pytest.raises(ValueError, match="Original error"),
            reproducibility.log_execution(script_path),
        ):
            raise ValueError("Original error")

    @patch.object(reproducibility, "get_db_connection")
    @patch.object(reproducibility, "get_git_commit")
    @patch.object(reproducibility, "get_script_id")
    def test_script_id_none_still_logs(
        self,
        mock_script_id: MagicMock,
        mock_git: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Test that execution is logged even when script_id is None."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        mock_git.return_value = "abc123"
        mock_script_id.return_value = None  # Script not in inventory

        script_path = reproducibility.PROJECT_ROOT / "scripts" / "new_script.py"

        with reproducibility.log_execution(script_path) as run_id:
            assert run_id is not None

        # Insert should still be called
        mock_cursor.execute.assert_called()


class TestProjectRoot:
    """Tests for PROJECT_ROOT constant."""

    def test_project_root_is_path(self) -> None:
        """Test that PROJECT_ROOT is a Path object."""
        assert isinstance(reproducibility.PROJECT_ROOT, Path)

    def test_project_root_is_resolved(self) -> None:
        """Test that PROJECT_ROOT is a resolved (absolute) path."""
        assert reproducibility.PROJECT_ROOT.is_absolute()


class TestDbName:
    """Tests for DB_NAME constant."""

    def test_db_name_value(self) -> None:
        """Test that DB_NAME has expected value."""
        assert reproducibility.DB_NAME == "cohort_projections_meta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
