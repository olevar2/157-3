#!/usr/bin/env node
/**
 * Platform3 Package Dependencies Updater
 * Updates package.json files with required dependencies
 */

const fs = require('fs').promises;
const path = require('path');

const requiredDependencies = {
    "uuid": "^9.0.0",
    "express": "^4.18.2"
};

async function updatePackageJsonFiles() {
    const servicesDir = path.join(__dirname, '../../services');
    
    try {
        const entries = await fs.readdir(servicesDir, { withFileTypes: true });
        
        for (const entry of entries) {
            if (entry.isDirectory()) {
                const packagePath = path.join(servicesDir, entry.name, 'package.json');
                await updatePackageJson(packagePath);
            }
        }
        
        console.log('Updated package.json files with required dependencies');
    } catch (error) {
        console.error('Failed to update package.json files:', error);
    }
}

async function updatePackageJson(packagePath) {
    try {
        const content = await fs.readFile(packagePath, 'utf-8');
        const packageData = JSON.parse(content);
        
        if (!packageData.dependencies) {
            packageData.dependencies = {};
        }
        
        let updated = false;
        for (const [dep, version] of Object.entries(requiredDependencies)) {
            if (!packageData.dependencies[dep]) {
                packageData.dependencies[dep] = version;
                updated = true;
            }
        }
        
        if (updated) {
            await fs.writeFile(packagePath, JSON.stringify(packageData, null, 2), 'utf-8');
            console.log(`Updated: ${packagePath}`);
        }
        
    } catch (error) {
        console.log(`Skipping ${packagePath} - file not found or invalid`);
    }
}

if (require.main === module) {
    updatePackageJsonFiles();
}
