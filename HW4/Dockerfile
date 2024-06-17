FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

ADD Pipfile.lock Pipfile /app/

WORKDIR /app

RUN pip install scikit-learn==1.5.0 pyarrow pandas

COPY starter.py /app/

CMD ["python", "starter.py", "2023", "5"]
