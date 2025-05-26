import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  ShowChart,
  AccountBalance,
  Assessment,
  Security,
  Settings,
  Person,
  Notifications,
  Logout,
  TrendingUp,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import { useAuth } from '../../contexts/AuthContext';
import LoadingScreen from '../Common/LoadingScreen';

const drawerWidth = 240;

interface NavigationItem {
  text: string;
  icon: React.ReactElement;
  path: string;
}

const navigationItems: NavigationItem[] = [
  { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
  { text: 'Trading', icon: <ShowChart />, path: '/trading' },
  { text: 'Portfolio', icon: <AccountBalance />, path: '/portfolio' },
  { text: 'Analytics', icon: <Assessment />, path: '/analytics' },
  { text: 'Risk Management', icon: <Security />, path: '/risk-management' },
  { text: 'Settings', icon: <Settings />, path: '/settings' },
  { text: 'Profile', icon: <Person />, path: '/profile' },
];

const Layout: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { user, logout, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (loading) {
    return <LoadingScreen />;
  }

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    handleProfileMenuClose();
    await logout();
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    setMobileOpen(false);
  };

  const drawer = (
    <Box>
      {/* Logo and Title */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
          color: 'white',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 40,
            height: 40,
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.2)',
            mr: 2,
          }}
        >
          <TrendingUp sx={{ fontSize: 24 }} />
        </Box>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Trading Platform
        </Typography>
      </Box>

      <Divider />

      {/* Navigation */}
      <List sx={{ pt: 2 }}>
        {navigationItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <motion.div
              key={item.text}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
            >
              <ListItem
                button
                onClick={() => handleNavigate(item.path)}
                sx={{
                  mx: 1,
                  mb: 0.5,
                  borderRadius: 2,
                  backgroundColor: isActive ? 'primary.main' : 'transparent',
                  color: isActive ? 'primary.contrastText' : 'text.primary',
                  '&:hover': {
                    backgroundColor: isActive ? 'primary.dark' : 'action.hover',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isActive ? 'primary.contrastText' : 'text.secondary',
                    minWidth: 40,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  primaryTypographyProps={{
                    fontWeight: isActive ? 600 : 400,
                  }}
                />
              </ListItem>
            </motion.div>
          );
        })}
      </List>

      <Divider sx={{ mt: 2 }} />

      {/* User info */}
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            p: 1,
            borderRadius: 2,
            background: 'rgba(33, 150, 243, 0.1)',
          }}
        >
          <Avatar
            sx={{
              width: 32,
              height: 32,
              mr: 1,
              bgcolor: 'primary.main',
              fontSize: '0.875rem',
            }}
          >
            {user?.name?.charAt(0) || 'U'}
          </Avatar>
          <Box sx={{ flexGrow: 1, minWidth: 0 }}>
            <Typography variant="body2" noWrap sx={{ fontWeight: 500 }}>
              {user?.name || 'Trading User'}
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap>
              Platform Owner
            </Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          background: 'rgba(26, 27, 35, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find(item => item.path === location.pathname)?.text || 'Dashboard'}
          </Typography>

          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton color="inherit" sx={{ mr: 1 }}>
              <Badge badgeContent={3} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* Profile Menu */}
          <Tooltip title="Account settings">
            <IconButton
              onClick={handleProfileMenuOpen}
              sx={{ ml: 1 }}
            >
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  bgcolor: 'primary.main',
                }}
              >
                {user?.name?.charAt(0) || 'U'}
              </Avatar>
            </IconButton>
          </Tooltip>

          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleProfileMenuClose}
            onClick={handleProfileMenuClose}
            PaperProps={{
              elevation: 8,
              sx: {
                overflow: 'visible',
                filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
                mt: 1.5,
                minWidth: 180,
                '& .MuiAvatar-root': {
                  width: 32,
                  height: 32,
                  ml: -0.5,
                  mr: 1,
                },
                '&:before': {
                  content: '""',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  right: 14,
                  width: 10,
                  height: 10,
                  bgcolor: 'background.paper',
                  transform: 'translateY(-50%) rotate(45deg)',
                  zIndex: 0,
                },
              },
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={() => navigate('/profile')}>
              <Avatar sx={{ bgcolor: 'primary.main' }}>
                {user?.name?.charAt(0) || 'U'}
              </Avatar>
              Profile
            </MenuItem>
            <Divider />
            <MenuItem onClick={() => navigate('/settings')}>
              <ListItemIcon>
                <Settings fontSize="small" />
              </ListItemIcon>
              Settings
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <Logout fontSize="small" />
              </ListItemIcon>
              Logout
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              background: 'linear-gradient(180deg, #1a1b23 0%, #141520 100%)',
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              background: 'linear-gradient(180deg, #1a1b23 0%, #141520 100%)',
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0b0d 0%, #1a1b23 100%)',
        }}
      >
        <Toolbar />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Outlet />
        </motion.div>
      </Box>
    </Box>
  );
};

export default Layout;
