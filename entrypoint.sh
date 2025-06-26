#!/usr/bin/env bash

uvicorn main:app --host 0.0.0.0 --port $PORT --log-config log_config.yaml