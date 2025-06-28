#!/usr/bin/env bash

exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-config log_config.yaml