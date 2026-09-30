FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY telemetry-service/ ./telemetry-service/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/telemetry-service ./telemetry-service

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/telemetry-service /telemetry-service
USER nonroot:nonroot
ENTRYPOINT ["/telemetry-service"]
