# offline_trainer.py
import os
import json
import datetime
import subprocess
import threading
import shutil
import signal
import time

class OfflineTrainerRunner:
    def __init__(self, here_dir, model_path, feedback_path, diag_path, min_samples=40,
                 max_concurrent_jobs=1, job_timeout_seconds=60*60):
        self.here = here_dir
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.diag_path = diag_path
        self.min_samples = int(min_samples)
        self._jobs = {}  # job_id -> info
        self._lock = threading.Lock()
        self.max_concurrent_jobs = int(max_concurrent_jobs)
        self.job_timeout_seconds = int(job_timeout_seconds)

    def _append_diag(self, entry):
        diagnostics = []
        try:
            if os.path.exists(self.diag_path):
                with open(self.diag_path, "r") as f:
                    diagnostics = json.load(f) or []
        except Exception:
            diagnostics = []
        diagnostics.append(entry)
        tmpd = self.diag_path + ".tmp"
        try:
            with open(tmpd, "w") as f:
                json.dump(diagnostics, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmpd, self.diag_path)
        except Exception:
            pass

    def _count_running(self):
        with self._lock:
            return sum(1 for v in self._jobs.values() if v.get("status") == "running")

    def start_job(self):
        # Check concurrency limit
        if self._count_running() >= self.max_concurrent_jobs:
            job_id = str(int(datetime.datetime.now().timestamp() * 1000))
            info = {"status": "rejected", "reason": "max_concurrent_jobs_reached", "started_at": datetime.datetime.now().isoformat()}
            with self._lock:
                self._jobs[job_id] = info
            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_rejected", "job_id": job_id})
            return job_id

        job_id = str(int(datetime.datetime.now().timestamp() * 1000))
        script = os.path.join(self.here, "offline_train_sgd.py")
        if not os.path.exists(script):
            entry = {"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_failed", "error": "script_not_found", "script": script}
            self._append_diag(entry)
            with self._lock:
                self._jobs[job_id] = {"status": "failed", "error": "script_not_found"}
            return job_id

        # Prepare env with paths and a log file
        env = os.environ.copy()
        env["MODEL_PATH"] = self.model_path
        env["FEEDBACK_PATH"] = self.feedback_path
        env["DIAG_PATH"] = self.diag_path
        env["MIN_SAMPLES"] = str(self.min_samples)

        logs_dir = os.path.join(os.path.dirname(self.diag_path), "train_logs")
        os.makedirs(logs_dir, exist_ok=True)
        stdout_path = os.path.join(logs_dir, f"{job_id}.out")
        stderr_path = os.path.join(logs_dir, f"{job_id}.err")

        try:
            proc = subprocess.Popen(
                ["python3", script],
                env=env,
                stdout=open(stdout_path, "wb"),
                stderr=open(stderr_path, "wb"),
                preexec_fn=os.setsid
            )
        except Exception as e:
            with self._lock:
                self._jobs[job_id] = {"status": "failed", "error": f"start_failed:{e}"}
            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_start_error", "job_id": job_id, "error": str(e)})
            return job_id

        info = {"status": "running", "started_at": datetime.datetime.now().isoformat(), "pid": proc.pid,
                "stdout": stdout_path, "stderr": stderr_path}
        with self._lock:
            self._jobs[job_id] = info

        self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_started", "job_id": job_id, "pid": proc.pid})

        # Monitor in background
        t = threading.Thread(target=self._monitor_proc, args=(job_id, proc, stdout_path, stderr_path), daemon=True)
        t.start()
        return job_id

    def _monitor_proc(self, job_id, proc, stdout_path, stderr_path):
        start = time.time()
        try:
            while True:
                if proc.poll() is not None:
                    break
                elapsed = time.time() - start
                if self.job_timeout_seconds and elapsed > self.job_timeout_seconds:
                    # timeout -> terminate process group
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time.sleep(2)
                        if proc.poll() is None:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    with self._lock:
                        entry = self._jobs.get(job_id, {})
                        entry.update({"status": "timeout", "finished_at": datetime.datetime.now().isoformat()})
                        self._jobs[job_id] = entry
                    self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_timeout", "job_id": job_id})
                    return
                time.sleep(1)

            ret = proc.returncode
            stdout_tail = self._tail_file(stdout_path, 2000)
            stderr_tail = self._tail_file(stderr_path, 2000)

            with self._lock:
                entry = self._jobs.get(job_id, {})
                entry.update({"returncode": ret, "finished_at": datetime.datetime.now().isoformat()})
                if ret == 0:
                    entry["status"] = "finished"
                else:
                    entry["status"] = "failed"
                entry["stdout_tail"] = stdout_tail
                entry["stderr_tail"] = stderr_tail
                self._jobs[job_id] = entry

            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_done", "job_id": job_id, "returncode": ret})
        except Exception as e:
            with self._lock:
                self._jobs[job_id] = {"status": "failed", "error": str(e)}
            self._append_diag({"timestamp": datetime.datetime.now().isoformat(), "type": "train_job_monitor_error", "job_id": job_id, "error": str(e)})

    def _tail_file(self, path, max_bytes=2000):
        try:
            if not os.path.exists(path):
                return ""
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                to_read = min(size, max_bytes)
                f.seek(size - to_read)
                data = f.read().decode("utf-8", errors="replace")
                return data
        except Exception:
            return ""

    def job_status(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self):
        with self._lock:
            return dict(self._jobs)

    def cancel_job(self, job_id):
        with self._lock:
            info = self._jobs.get(job_id)
            if not info or info.get("status") != "running":
                return False
            pid = info.get("pid")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1)
            return True
        except Exception:
            return False
