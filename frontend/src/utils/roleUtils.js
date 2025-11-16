/**
 * Role utility functions for checking user permissions
 */

/**
 * Check if a user has a specific role
 * @param {Object} user - User object with roles array
 * @param {string|number} roleNameOrId - Role name (string) or role ID (number)
 * @returns {boolean}
 */
export function hasRole(user, roleNameOrId) {
  if (!user || !user.roles || !Array.isArray(user.roles)) {
    return false;
  }
  
  // Check by role name (string) or role ID (number)
  return user.roles.some(role => {
    if (typeof roleNameOrId === 'string') {
      return role.name === roleNameOrId || role.name?.toLowerCase() === roleNameOrId.toLowerCase();
    } else {
      return role.id === roleNameOrId;
    }
  });
}

/**
 * Check if a user has any of the specified roles
 * @param {Object} user - User object with roles array
 * @param {Array<string|number>} roleNamesOrIds - Array of role names or IDs
 * @returns {boolean}
 */
export function hasAnyRole(user, roleNamesOrIds) {
  if (!roleNamesOrIds || roleNamesOrIds.length === 0) {
    return true; // No restrictions
  }
  
  return roleNamesOrIds.some(roleNameOrId => hasRole(user, roleNameOrId));
}

/**
 * Check if a user has all of the specified roles
 * @param {Object} user - User object with roles array
 * @param {Array<string|number>} roleNamesOrIds - Array of role names or IDs
 * @returns {boolean}
 */
export function hasAllRoles(user, roleNamesOrIds) {
  if (!roleNamesOrIds || roleNamesOrIds.length === 0) {
    return true; // No restrictions
  }
  
  return roleNamesOrIds.every(roleNameOrId => hasRole(user, roleNameOrId));
}

/**
 * Get user's role names as an array
 * @param {Object} user - User object with roles array
 * @returns {Array<string>}
 */
export function getUserRoleNames(user) {
  if (!user || !user.roles || !Array.isArray(user.roles)) {
    return [];
  }
  return user.roles.map(role => role.name).filter(Boolean);
}

/**
 * Get role instructions for a user (from their assigned roles)
 * @param {Object} user - User object with roles array
 * @returns {Array<string>} Array of AI instructions from all user's roles
 */
export function getUserRoleInstructions(user) {
  if (!user || !user.roles || !Array.isArray(user.roles)) {
    return [];
  }
  return user.roles
    .map(role => role.ai_instructions)
    .filter(Boolean);
}

