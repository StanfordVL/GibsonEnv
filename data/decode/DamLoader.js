/**
 * Customized DamLoader
 * Input:  .dam input file path, .obj output path
 * Output: boolean, whether loading succeed. If succeed,
 *         .obj file will be stored at given path
 */

var THREE     = require("./Three");
var PROTOBUF  = require("./Protobuf.js");
var fs        = require("fs");
require('./helpers.js')

var LOW  = 'LOW'
var MID  = 'MID'
var HIGH = 'HIGH'


function DamLoader(inputDBPath, outputDBPath, quality=HIGH) {
    this.inputDB  = inputDBPath,
    this.outputDB = outputDBPath,
    this.quality  = quality,
    this.builder  = PROTOBUF.loadProto(protoToken),
	this.decoder  = this.builder.build("binary_mesh"),
    this.mtlName  = "complete.mtl"
}


DamLoader.prototype = {
    constructor: DamLoader,
    load: function(id, mtlName, meshUrl) {
        self = this;
        var damPath = self.inputDB + '/' + id + '/dam';
        fs.readFile(self.selectDam(damPath), function read(err, data) {
            if (err) {
                throw err;
            }

            // Invoke the next step here however you like
            // console.log(content);   // Put all of the code here (not the best solution)
            self.parse(data, id, meshUrl)
        });

        if (mtlName) {
            this.mtlName = mtlName;
        }

    },
    selectDam: function(dir) {
        var selected;
        var allFiles = [];
        var files = fs.readdirSync(dir)
        files.forEach(file => {
            allFiles.push(file)
        });
        if (this.quality == LOW) {
            selected = allFiles.filter(function(f) {
                return f.endsWith('_10k.dam')
            })
        } else if (this.quality == MID) {
            selected = allFiles.filter(function(f) {
                return f.endsWith('_50k.dam')
            })
        } else {
            // High quality
            selected = allFiles.filter(function(f) {
                return (!f.endsWith('v2.dam')) && (!f.endsWith('_10k.dam')) 
                && (!f.endsWith('_50k.dam')) && (f !== 'tesselate.dam') && f.endsWith('.dam')
            })
        }
        return dir + '/' + selected
    },
    parse: function(e, id, t) {
        var o = this.readProtobuf(e);
        try {
            // c.time("convert to webgl"),
            this.convertProtobufToSceneObject(o, id, t)
            // c.timeEnd("convert to webgl")
        } catch (e) {
        	console.log("failed parsing .dam");
        	throw new Error();
            // return c.error("failed parsing .dam"),
            // c.error(e.message),
        }
    },
    readProtobuf: function(e) {
        var t;
        try {
            // c.time("parse proto"),
            t = this.decoder.decode(e)
            // c.timeEnd("parse proto")
        } catch (e) {
        	console.log("failed parsing proto for .dam");
        	throw new Error();
            // return c.error("failed parsing proto for .dam"),
            // c.error(e.message),
            // null
        }
        return t
    },
    convertProtobufToSceneObject: function(e, id, t) {
        self = this;
        function a(e) {
            var i = new THREE.BufferGeometry;
            return i.addAttribute("position", new THREE.BufferAttribute(new Float32Array(e.vertices.xyz,0,3),3)),
            e.vertices.uv.length > 0 && i.addAttribute("uv", new THREE.BufferAttribute(new Float32Array(e.vertices.uv,0,2),2)),
            i.setIndex(new THREE.BufferAttribute(new Uint32Array(e.faces.faces,0,1),1)),
            i.applyMatrix(s),
            i.computeBoundingBox(),
            new CHUNK({
                geometry: i,
                textureName: e.material_name,
                name: e.chunk_name,
                meshUrl: t
            })
        }

        if (0 === e.chunk.length) {
        	console.log("No chunks in damfile...");
            return ;
        }
        var s = new THREE.Matrix4;
        s.set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
        var l = e.chunk.map(a);

        // 3D Model Scraping & Saving
        var exporter = new THREE.OBJExporter();

        if (document.mergeModels) {
        	for (var index = 0; index < l.length; index++) {
        		l[0].geometry.merge(l[index].geometry);
        	}
        	self.saveFile(exporter.parse(l[0]), id, 
                "out_res.obj");
            
            console.log("Success (merged): " + id);

        } else {
			// for (var index = 0; index < l.length; index++) {
        	//	self.saveFile(exporter.parse(l[index]), id, 
            //        id + "_" + self.quality + "_" + index + ".obj");
        	//}
            this.mtlName = id + '.mtl';  // need to be commented out for Gates 2nd floor
            self.saveFile(exporter.parse(l, this.mtlName), id, 
                "out_res.obj");
            self.saveFile(exporter.generateMtl(l), id, this.mtlName);
        };

        if (l) {
        	return ;
        } else {
        	console.log(".dam protobuf came out with no chunks...")
        	throw new Error()
        }

    },
    saveFile: function(content, id, name) {
        if (self.outputDB !== "") {
            if (!fs.existsSync(self.outputDB + '/' + id)){
                fs.mkdirSync(self.outputDB + '/' + id);
            }
            var filePath = self.outputDB + '/' + id + '/modeldata/' + name
            fs.writeFileSync(filePath, content)
        }
    }
}


module.exports = DamLoader