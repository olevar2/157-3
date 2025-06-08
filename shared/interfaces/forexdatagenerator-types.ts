/**
 * TypeScript interfaces for ForexDataGenerator
 * Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for Forexdatagenerator.calculate
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorCalculateRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
    /** data parameter */
    data?: Platform3Types.PriceData[];
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for Forexdatagenerator.calculate
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorCalculateResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: Platform3Types.IndicatorResult | Platform3Types.IndicatorResult[];
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

/**
 * Request interface for Forexdatagenerator.get_current_value
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorGet_Current_ValueRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for Forexdatagenerator.get_current_value
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorGet_Current_ValueResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: any;
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

/**
 * Request interface for Forexdatagenerator.reset
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorResetRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for Forexdatagenerator.reset
 * @description Generated from ai-platform/ai-services/ml-service/src/data/ForexDataGenerator.py
 */
export interface ForexdatageneratorResetResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: any;
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

export default Forexdatagenerator;
