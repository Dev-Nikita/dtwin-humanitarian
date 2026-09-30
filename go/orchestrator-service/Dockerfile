FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY orchestrator-service/ ./orchestrator-service/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/orchestrator-service ./orchestrator-service

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/orchestrator-service /orchestrator-service
USER nonroot:nonroot
ENTRYPOINT ["/orchestrator-service"]
