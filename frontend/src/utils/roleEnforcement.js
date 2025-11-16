/**
 * Role enforcement utilities - Parse role instructions and enforce permissions
 */

/**
 * Parse role instructions to extract permissions
 * Looks for keywords like "can", "cannot", "access", "restrict", "grants", "adds", etc.
 */
export function parseRolePermissions(roleInstructions) {
  if (!roleInstructions || typeof roleInstructions !== 'string') {
    return {
      grantedFeatures: [], // Features this role grants (adds new features)
      allowedFeatures: [], // Features this role allows (explicit permission)
      deniedFeatures: [], // Features this role denies
      customRules: [],
    };
  }

  const instructions = roleInstructions.toLowerCase();
  const grantedFeatures = []; // New features granted by this role
  const allowedFeatures = []; // Features explicitly allowed
  const deniedFeatures = []; // Features denied
  const customRules = [];

  // Feature keywords mapping
  const featureKeywords = {
    'messages': 'messages',
    'messaging': 'messages',
    'chat': 'messages',
    'members': 'members',
    'videos': 'videos',
    'video': 'videos',
    'paint': 'paint',
    'painting': 'paint',
    'drawing': 'paint',
    'todos': 'todos',
    'todo': 'todos',
    'todo list': 'todos',
    'ai chat': 'ai-chat',
    'ai': 'ai-chat',
    'docs': 'docs',
    'documents': 'docs',
    'images': 'docs',
    'blog': 'blog',
    'blogs': 'blog',
    'cloud pc': 'cloud-pcs',
    'cloud pcs': 'cloud-pcs',
    'cloudpc': 'cloud-pcs',
    'cloudpcs': 'cloud-pcs',
    'virtual pc': 'cloud-pcs',
    'virtual machine': 'cloud-pcs',
    'vm': 'cloud-pcs',
    'admin': 'admin',
    'administration': 'admin',
  };

  // Check for granted features (new features added by role)
  // Patterns like "grants access to", "adds feature", "provides access to", "enables access to"
  for (const [keyword, feature] of Object.entries(featureKeywords)) {
    if (
      instructions.includes(`grants ${keyword}`) ||
      instructions.includes(`grants access to ${keyword}`) ||
      instructions.includes(`adds ${keyword}`) ||
      instructions.includes(`adds access to ${keyword}`) ||
      instructions.includes(`provides ${keyword}`) ||
      instructions.includes(`provides access to ${keyword}`) ||
      instructions.includes(`enables ${keyword}`) ||
      instructions.includes(`enables access to ${keyword}`) ||
      instructions.includes(`gives ${keyword}`) ||
      instructions.includes(`gives access to ${keyword}`) ||
      instructions.includes(`unlocks ${keyword}`) ||
      instructions.includes(`unlocks access to ${keyword}`)
    ) {
      if (!grantedFeatures.includes(feature)) {
        grantedFeatures.push(feature);
      }
      // Also add to allowed features
      if (!allowedFeatures.includes(feature)) {
        allowedFeatures.push(feature);
      }
    }
  }

  // Check for allowed features (explicit permission)
  for (const [keyword, feature] of Object.entries(featureKeywords)) {
    // Patterns like "can access messages", "allows videos", "has access to paint"
    if (
      instructions.includes(`can ${keyword}`) ||
      instructions.includes(`can access ${keyword}`) ||
      instructions.includes(`allows ${keyword}`) ||
      instructions.includes(`has access to ${keyword}`) ||
      instructions.includes(`can use ${keyword}`)
    ) {
      // Only add if not already granted (to avoid duplicates)
      if (!grantedFeatures.includes(feature) && !allowedFeatures.includes(feature)) {
        allowedFeatures.push(feature);
      }
    }

    // Patterns like "cannot access messages", "restricts videos", "no access to paint"
    if (
      instructions.includes(`cannot ${keyword}`) ||
      instructions.includes(`can't ${keyword}`) ||
      instructions.includes(`restricts ${keyword}`) ||
      instructions.includes(`no access to ${keyword}`) ||
      instructions.includes(`denies ${keyword}`) ||
      instructions.includes(`blocks ${keyword}`) ||
      instructions.includes(`prohibits ${keyword}`) ||
      instructions.includes(`disallows ${keyword}`)
    ) {
      if (!deniedFeatures.includes(feature)) {
        deniedFeatures.push(feature);
      }
    }
  }

  // Check for "only" restrictions (e.g., "only access blog")
  const onlyMatch = instructions.match(/only (?:access|use|view|see|allows|grants) ([^.,]+)/);
  if (onlyMatch) {
    const onlyFeature = onlyMatch[1].trim();
    for (const [keyword, feature] of Object.entries(featureKeywords)) {
      if (onlyFeature.includes(keyword)) {
        // Deny all features except this one
        Object.values(featureKeywords).forEach(f => {
          if (f !== feature && !deniedFeatures.includes(f)) {
            deniedFeatures.push(f);
          }
        });
        if (!allowedFeatures.includes(feature)) {
          allowedFeatures.push(feature);
        }
        if (!grantedFeatures.includes(feature)) {
          grantedFeatures.push(feature);
        }
      }
    }
  }

  return {
    grantedFeatures, // New features granted by this role
    allowedFeatures, // Features explicitly allowed
    deniedFeatures, // Features denied
    customRules: [instructions], // Store full instructions for custom logic
  };
}

/**
 * Get all permissions for a user based on their roles
 */
export function getUserPermissions(user) {
  if (!user || !user.roles || !Array.isArray(user.roles)) {
    return {
      grantedFeatures: [],
      allowedFeatures: [],
      deniedFeatures: [],
      customRules: [],
    };
  }

  const allGranted = new Set(); // Features granted by roles (new features)
  const allAllowed = new Set(); // Features explicitly allowed
  const allDenied = new Set(); // Features denied
  const allCustomRules = [];

  user.roles.forEach(role => {
    const permissions = parseRolePermissions(role.ai_instructions);
    permissions.grantedFeatures.forEach(f => allGranted.add(f));
    permissions.allowedFeatures.forEach(f => allAllowed.add(f));
    permissions.deniedFeatures.forEach(f => allDenied.add(f));
    allCustomRules.push(...permissions.customRules);
  });

  // Denied features override allowed/granted features
  const finalGranted = Array.from(allGranted).filter(f => !allDenied.has(f));
  const finalAllowed = Array.from(allAllowed).filter(f => !allDenied.has(f));

  return {
    grantedFeatures: finalGranted, // New features added by roles
    allowedFeatures: finalAllowed, // Features explicitly allowed
    deniedFeatures: Array.from(allDenied), // Features denied
    customRules: allCustomRules,
  };
}

/**
 * Check if a user can access a specific feature
 */
export function canAccessFeature(user, featureName) {
  if (!user) return false;

  // Admins can access everything
  if (user.is_admin) return true;

  const permissions = getUserPermissions(user);

  // If user has no roles, allow access (default behavior)
  if (user.roles && user.roles.length === 0) {
    return true;
  }

  // If there are denied features, check those first (denied overrides everything)
  if (permissions.deniedFeatures.length > 0) {
    if (permissions.deniedFeatures.includes(featureName)) {
      return false;
    }
  }

  // Check if feature is granted by a role (new features added by roles)
  if (permissions.grantedFeatures.length > 0) {
    if (permissions.grantedFeatures.includes(featureName)) {
      return true; // Role grants this feature
    }
  }

  // If there are allowed features, user must be in that list
  if (permissions.allowedFeatures.length > 0) {
    return permissions.allowedFeatures.includes(featureName);
  }

  // If roles have specific permissions but feature isn't granted/allowed, check if it's denied
  // If not explicitly denied and roles exist, allow access (roles add permissions, don't restrict by default)
  // But if roles have any permissions defined, we should check them
  if (permissions.grantedFeatures.length > 0 || permissions.allowedFeatures.length > 0 || permissions.deniedFeatures.length > 0) {
    // Roles are defined - if feature is not granted/allowed and not denied, allow it (default behavior)
    return true;
  }

  // If no specific permissions, allow access (default)
  return true;
}

/**
 * Get list of features user can access
 */
export function getAccessibleFeatures(user) {
  if (!user) return [];

  if (user.is_admin) {
    // Admins can access everything
    return ['messages', 'members', 'videos', 'paint', 'todos', 'ai-chat', 'docs', 'blog', 'cloud-pcs', 'admin'];
  }

  const permissions = getUserPermissions(user);
  const defaultFeatures = ['messages', 'members', 'videos', 'paint', 'todos', 'ai-chat', 'docs', 'blog', 'cloud-pcs'];

  // If user has no roles, return all default features
  if (!user.roles || user.roles.length === 0) {
    return defaultFeatures;
  }

  // Start with default features
  const accessibleFeatures = new Set(defaultFeatures);

  // Add granted features (new features added by roles)
  permissions.grantedFeatures.forEach(f => accessibleFeatures.add(f));

  // Add allowed features
  permissions.allowedFeatures.forEach(f => accessibleFeatures.add(f));

  // Remove denied features (denied overrides everything)
  permissions.deniedFeatures.forEach(f => accessibleFeatures.delete(f));

  // If roles have specific permissions but no granted/allowed features, check denied
  // If there are denied features, return default minus denied
  if (permissions.deniedFeatures.length > 0 && permissions.grantedFeatures.length === 0 && permissions.allowedFeatures.length === 0) {
    return defaultFeatures.filter(f => !permissions.deniedFeatures.includes(f));
  }

  return Array.from(accessibleFeatures);
}

