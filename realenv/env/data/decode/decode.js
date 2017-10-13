var DamLoader = require("./DamLoader.js")
var argv = require('optimist').argv

document = {};
document.mergeModels = false;

rootDir   = argv.rootdir
modelId   = argv.model

loader = new DamLoader(rootDir, rootDir)

loader.load(modelId)