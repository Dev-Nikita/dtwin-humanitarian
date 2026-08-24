FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY digital-twin-service/ ./digital-twin-service/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/digital-twin-service ./digital-twin-service

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/digital-twin-service /digital-twin-service
USER nonroot:nonroot
ENTRYPOINT ["/digital-twin-service"]
