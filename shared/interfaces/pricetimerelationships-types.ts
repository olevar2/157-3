/**
 * TypeScript interfaces for price_time_relationships
 * Generated from engines/gann/price_time_relationships.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for PriceTimeRelationships.calculate
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsCalculateRequest {
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
 * Response interface for PriceTimeRelationships.calculate
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsCalculateResponse extends Platform3Types.ApiResponse {
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
 * Request interface for PriceTimeRelationships.get_current_value
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsGet_Current_ValueRequest {
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
 * Response interface for PriceTimeRelationships.get_current_value
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsGet_Current_ValueResponse extends Platform3Types.ApiResponse {
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
 * Request interface for PriceTimeRelationships.reset
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsResetRequest {
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
 * Response interface for PriceTimeRelationships.reset
 * @description Generated from engines/gann/price_time_relationships.py
 */
export interface PriceTimeRelationshipsResetResponse extends Platform3Types.ApiResponse {
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

export default PriceTimeRelationships;
