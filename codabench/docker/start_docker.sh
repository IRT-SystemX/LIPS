# benchmark tested
BENCHMARK_IDS=(1 2 3)

# start docker container in background
docker-compose up -d

# get container name
CONTAINER_NAME=`docker ps --format "{{.Names}}"`

# copy relevant file into the container
docker exec -it $CONTAINER_NAME mkdir -p /app/program
docker cp ../code_submission/ $CONTAINER_NAME:/app/ingested_program
docker cp ../ingestion_program/ $CONTAINER_NAME:/app/program/ingestion_program
docker cp ../scoring_program/ $CONTAINER_NAME:/app/program/scoring_program
for BENCHMARK_ID in "${BENCHMARK_IDS[@]}"
  do
    docker cp ../"ingestion_program_$BENCHMARK_ID"/ $CONTAINER_NAME:/app/program/"ingestion_$BENCHMARK_ID"
    docker cp ../"scoring_program_$BENCHMARK_ID"/ $CONTAINER_NAME:/app/program/"scoring_$BENCHMARK_ID"
  done


# creates scripts to run ingestion and scoring; those scripts take as argument the ID of the benchmark
echo "python3 program/ingestion_\${1:-1}/ingestion.py --ingestion_program_dir program/ingestion_\${1:-1} --code_dir ingested_program --benchmark_output_dir /app/benchmark_output_\${1:-1}" > ingestion.sh ; chmod a+x ingestion.sh; docker cp ingestion.sh $CONTAINER_NAME:/app ; rm ingestion.sh
echo "python3 program/scoring_\${1:-1}/score.py --scoring_program_dir program/scoring_\${1:-1} --benchmark_output_dir /app/benchmark_output_\${1:-1} --score_dir /app/output_\${1:-1}" > scoring.sh ; chmod a+x scoring.sh; docker cp scoring.sh $CONTAINER_NAME:/app ; rm scoring.sh

# go into the container
docker exec -it $CONTAINER_NAME bash