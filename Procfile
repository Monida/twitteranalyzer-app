redisweb: gunicorn app:app --log-file=-
worker: python worker.py
heroku ps:scale redisweb=1
heroku ps:scale worker=1
heroku run worker
