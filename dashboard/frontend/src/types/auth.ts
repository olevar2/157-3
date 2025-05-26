export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  phone?: string;
  country?: string;
  timezone: string;
  language: string;
  status: 'pending' | 'active' | 'suspended' | 'closed';
  kycStatus: string;
  twoFactorEnabled: boolean;
  emailVerified: boolean;
  phoneVerified: boolean;
  createdAt: string;
  updatedAt: string;
  lastLogin?: string;
  preferences?: any;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  refreshToken?: string;
  user: User;
  sessionId?: string;
}

export interface RegisterData {
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
  phone?: string;
  country?: string;
}

export interface AuthError {
  message: string;
  code?: string;
  field?: string;
}
