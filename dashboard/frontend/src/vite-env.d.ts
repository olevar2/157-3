/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_MARKET_DATA_URL: string
  readonly VITE_TRADING_SERVICE_URL: string
  readonly VITE_USER_SERVICE_URL: string
  readonly VITE_WS_URL: string
  readonly VITE_WS_MARKET_DATA_URL: string
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_DEV_MODE: string
  readonly VITE_MOCK_DATA: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
