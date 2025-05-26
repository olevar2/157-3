import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';

import { authAPI } from '../services/api';
import { User, LoginCredentials, LoginResponse } from '../types/auth';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const queryClient = useQueryClient();

  // Check if user is authenticated on app load
  const { data: currentUser, isLoading: userLoading } = useQuery(
    'currentUser',
    authAPI.getCurrentUser,
    {
      retry: false,
      enabled: !!localStorage.getItem('token'),
      onSuccess: (userData) => {
        setUser(userData);
        setLoading(false);
      },
      onError: () => {
        localStorage.removeItem('token');
        setUser(null);
        setLoading(false);
      },
    }
  );

  // Login mutation
  const loginMutation = useMutation(authAPI.login, {
    onSuccess: (response: LoginResponse) => {
      localStorage.setItem('token', response.token);
      setUser(response.user);
      queryClient.setQueryData('currentUser', response.user);
      toast.success('Welcome back!');
    },
    onError: (error: any) => {
      const message = error.response?.data?.message || 'Login failed';
      toast.error(message);
      throw error;
    },
  });

  // Logout mutation
  const logoutMutation = useMutation(authAPI.logout, {
    onSuccess: () => {
      localStorage.removeItem('token');
      setUser(null);
      queryClient.clear();
      toast.success('Logged out successfully');
    },
    onError: () => {
      // Even if logout fails on server, clear local data
      localStorage.removeItem('token');
      setUser(null);
      queryClient.clear();
    },
  });

  // Update profile mutation
  const updateProfileMutation = useMutation(authAPI.updateProfile, {
    onSuccess: (updatedUser) => {
      setUser(updatedUser);
      queryClient.setQueryData('currentUser', updatedUser);
      toast.success('Profile updated successfully');
    },
    onError: (error: any) => {
      const message = error.response?.data?.message || 'Failed to update profile';
      toast.error(message);
      throw error;
    },
  });

  // Check authentication status on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setLoading(false);
    }
  }, []);

  // Update loading state based on user query
  useEffect(() => {
    if (!userLoading) {
      setLoading(false);
    }
  }, [userLoading]);

  const login = async (credentials: LoginCredentials) => {
    await loginMutation.mutateAsync(credentials);
  };

  const logout = async () => {
    await logoutMutation.mutateAsync();
  };

  const updateProfile = async (data: Partial<User>) => {
    await updateProfileMutation.mutateAsync(data);
  };

  const refresh = async () => {
    try {
      const userData = await authAPI.getCurrentUser();
      setUser(userData);
      queryClient.setQueryData('currentUser', userData);
    } catch (error) {
      localStorage.removeItem('token');
      setUser(null);
      queryClient.clear();
      throw error;
    }
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    loading: loading || loginMutation.isLoading || logoutMutation.isLoading,
    login,
    logout,
    updateProfile,
    refresh,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
