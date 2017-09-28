var THREE     = require("./Three");
var PROTOBUF  = require("./Protobuf.js");
var fs        = require("fs");
var buffer    = require("buffer");



var CoreViewModes = {
    PANORAMA: "panorama",
    DOLLHOUSE: "dollhouse",
    FLOORPLAN: "floorplan"
}


var Viewmode = {
    MESH: "mesh",
    OUTDOOR: "outdoor",
    TRANSITIONING: "transitioning",
    toInt: function(e) {
        switch (e) {
        case this.PANORAMA:
            return 1;
        case this.DOLLHOUSE:
            return 2;
        case this.FLOORPLAN:
            return 3;
        case this.OUTDOOR:
            return 4;
        case this.TRANSITIONING:
            return -1
        }
    },
    fromInt: function(e) {
        switch (e) {
        case "1":
        case 1:
            return this.PANORAMA;
        case "2":
        case 2:
            return this.DOLLHOUSE;
        case "3":
        case 3:
            return this.FLOORPLAN;
        case "4":
        case 4:
            return this.OUTDOOR
        }
    },
    convertWorkshopModeInt: function(e) {
        switch (e) {
        case "0":
        case 0:
            return this.PANORAMA;
        case "1":
        case 1:
            return this.FLOORPLAN;
        case "2":
        case 2:
            return this.DOLLHOUSE;
        case "3":
        case 3:
            return this.MESH
        }
    }
};





var COLORS = {
    newBlue: new THREE.Color(4967932),
    altBlue: new THREE.Color(47355),
    classicBlue: new THREE.Color(53759),
    mpYellow: new THREE.Color(16502016),
    mpOrange: new THREE.Color(16428055),
    mpBlue: new THREE.Color(12096),
    mpLtGrey: new THREE.Color(13751252),
    mpDkGrey: new THREE.Color(10000019),
    mpRed: new THREE.Color(12525854),
    mpOrangeDesat: new THREE.Color(16764529),
    mpBlueDesat: new THREE.Color(4034734),
    mpRedDesat: new THREE.Color(14705505),
    white: new THREE.Color(16777215),
    black: new THREE.Color(0),
    _desat: function(e, t) {
        var i = t || .3
          , r = (new THREE.Color).copy(e).getHSL();
        return (new THREE.Color).setHSL(r.h, r.s * (1 - i), r.l)
    },
    _darken: function(e, t) {
        var i = t || .2
          , r = (new THREE.Color).copy(e).getHSL();
        return (new THREE.Color).setHSL(r.h, r.s, r.l * (1 - i))
    }
}


var o = "precision highp float;\nprecision highp int;\n\nuniform mat4 modelMatrix;\nuniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\nuniform mat4 viewMatrix;\nuniform mat3 normalMatrix;\nuniform vec3 cameraPosition;\nattribute vec3 position;\nattribute vec3 normal;\nattribute vec2 uv;\n"
var a = "precision highp float;\nprecision highp int;\n\nuniform mat4 viewMatrix;\nuniform vec3 cameraPosition;\n";

var SHADERS = {
    basicTextured: {
        uniforms: {
            tDiffuse: {
                type: "t",
                value: null
            },
            alpha: {
                type: "f",
                value: 1
            }
        },
        vertexShader: "varying vec2 vUv;\nvoid main() {\n  vUv = uv;\n  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n}",
        fragmentShader: "varying vec2 vUv;\nuniform float alpha;\nuniform sampler2D tDiffuse;\nvoid main() {\n  vec4 texColor = texture2D(tDiffuse, vUv);\n  gl_FragColor = vec4(texColor.rgb, texColor.a * alpha);\n}"
    },
    copyCubeMap: {
        uniforms: {
            tDiffuse: {
                type: "t",
                value: null
            },
            alpha: {
                type: "f",
                value: 1
            }
        },
        vertexShader: "varying vec3 vWorldPos;\nvoid main() {\n  vWorldPos = vec3(-position.x, -position.y, position.z);\n  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n}",
        fragmentShader: "varying vec3 vWorldPos;\nuniform float alpha;\nuniform samplerCube tDiffuse;\nvoid main() {\n  vec4 texColor = textureCube(tDiffuse, vWorldPos);\n  gl_FragColor = vec4(texColor.rgb, texColor.a * alpha);\n}"
    },
    cube: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            opacity: {
                type: "f",
                value: 1
            }
        },
        vertexShader: o + "varying vec3 vWorldPosition;\n\nvoid main() {\n  vWorldPosition = position;\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}\n",
        fragmentShader: a + "uniform samplerCube map;\nuniform float opacity;\n\nvarying vec3 vWorldPosition;\n\nvoid main() {\n  vec4 color = textureCube( map, vec3( -vWorldPosition.x, vWorldPosition.yz ) );\n  gl_FragColor = vec4(color.rgb, opacity);\n}\n"
    },
    model: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            modelAlpha: {
                type: "f",
                value: 1 // r.modelAlpha
            },
            opacity: {
                type: "f",
                value: 1
            },
            progress: {
                type: "f",
                value: 0
            },
            blackout: {
                type: "i",
                value: 0
            },
            pano0Map: {
                type: "t",
                value: null
            },
            pano0Position: {
                type: "v3",
                value: new THREE.Vector3
            },
            pano0Matrix: {
                type: "m4",
                value: new THREE.Matrix4
            },
            pano1Map: {
                type: "t",
                value: null
            },
            pano1Position: {
                type: "v3",
                value: new THREE.Vector3
            },
            pano1Matrix: {
                type: "m4",
                value: new THREE.Matrix4
            }
        },
        vertexShader: o + "uniform vec3 pano0Position;\nuniform mat4 pano0Matrix;\n\nuniform vec3 pano1Position;\nuniform mat4 pano1Matrix;\n\nvarying vec2 vUv;\nvarying vec3 vWorldPosition0;\nvarying vec3 vWorldPosition1;\n\nvoid main() {\n\n  vUv = uv;\n  vec4 worldPosition = modelMatrix * vec4(position, 1.0);\n\n  vec3 positionLocalToPanoCenter0 = worldPosition.xyz - pano0Position;\n  vWorldPosition0 = (vec4(positionLocalToPanoCenter0, 1.0) * pano0Matrix).xyz;\n  vWorldPosition0.x *= -1.0;\n\n  vec3 positionLocalToPanoCenter1 = worldPosition.xyz - pano1Position;\n  vWorldPosition1 = (vec4(positionLocalToPanoCenter1, 1.0) * pano1Matrix).xyz;\n  vWorldPosition1.x *= -1.0;\n\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n}\n",
        fragmentShader: a + "uniform sampler2D map;\nuniform float modelAlpha;\nuniform float opacity;\nuniform float progress;\nuniform int blackout;\n\nuniform vec3 pano0Position;\nuniform samplerCube pano0Map;\n\nuniform vec3 pano1Position;\nuniform samplerCube pano1Map;\n\nvarying vec2 vUv;\nvarying vec3 vWorldPosition0;\nvarying vec3 vWorldPosition1;\n\nvoid main() {\n\tconst vec4 BLACK = vec4(0.0, 0.0, 0.0, 1.0);\n\tconst vec4 GREY  = vec4(0.5, 0.5, 0.5, 1.0);\n\n\tvec4 colorFromPanos;\n\tvec4 colorFromPano0 = textureCube( pano0Map, vWorldPosition0.xyz);\n\tvec4 colorFromPano1 = textureCube( pano1Map, vWorldPosition1.xyz);\n\n\tif (blackout == 0) {\n\t\tcolorFromPanos = mix(colorFromPano0, colorFromPano1, progress);\n\t} else if (blackout == 1) {\n\t\tcolorFromPanos = mix(colorFromPano0, BLACK, min(1.0, progress*2.0));\n\t\tcolorFromPanos = mix(colorFromPanos, colorFromPano1, max(0.0, progress * 2.0 - 1.0));\n\t} else if (blackout == 2) {\n\t\tcolorFromPanos = mix(colorFromPano0, BLACK, progress);\n\t} else if (blackout == 3) {\n\t\tcolorFromPanos = mix(BLACK, colorFromPano1, max(0.0, progress * 2.0 - 1.0));\n\t} \n\n\tvec4 colorFromTexture = texture2D( map, vUv );\n\tcolorFromPanos = mix(colorFromPanos, colorFromTexture, modelAlpha);\n\n\tfloat whiteness = 1.0 - smoothstep(0.1, 0.2, opacity);\n\tcolorFromPanos = mix(colorFromPanos, GREY, whiteness);\n\tgl_FragColor = vec4(colorFromPanos.rgb, opacity);\n}\n"
    },
    modelOutside: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            opacity: {
                type: "f",
                value: 1
            }
        },
        vertexShader: o + "varying vec2 vUv;\n\nvoid main() {\n\n  vUv = uv;\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n}\n",
        fragmentShader: a + "uniform sampler2D map;\nuniform float opacity;\nvarying vec2 vUv;\n\nvec4 white = vec4(0.5, 0.5, 0.5, 1.0);\n\nvoid main() {\n\n  vec4 colorFromTexture = texture2D( map, vUv );\n  float whiteness = 1.0 - smoothstep(0.1, 0.2, opacity);\n  colorFromTexture = mix(colorFromTexture, white, whiteness);\n  gl_FragColor = vec4(colorFromTexture.rgb, opacity);\n\n}\n"
    },
    ribbon: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            opacity: {
                type: "f",
                value: 1
            },
            color: {
                type: "c",
                value: new THREE.Color(COLORS.newBlue) // r.path.color)
            }
        },
        vertexShader: o + "varying vec2 vUv;\nvarying vec3 vN;\nvarying vec4 vP;\n\nvoid main() {\n\n  vUv = uv;\n  vN= normalMatrix * normal;\n  vP = modelViewMatrix * vec4( position, 1.0 );\n  gl_Position = projectionMatrix * vP;\n}\n",
        fragmentShader: a + "uniform sampler2D map;\nuniform float opacity;\nvarying vec2 vUv;\nuniform vec3 color;\nvarying vec3 vN; // show-1182\nvarying vec4 vP; // show-1182\n\nvoid main() {\n\t// TODO add scroll-in and pulsing behaviors\n\tvec3 vNn = normalize(vN);\n\tvec3 vPn = normalize(vP.xyz);\n\tfloat f = pow(1.0-abs(dot(vNn,vPn)),0.2);\n  vec4 colorFromTexture = texture2D( map, vUv );\n  colorFromTexture.a *= f;\n  gl_FragColor = vec4((color.rgb*colorFromTexture.rgb),\n  \t\t\t\t\t\t(opacity*colorFromTexture.a));\n}\n"
    },
    waypoint: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            opacity: {
                type: "f",
                value: 1
            },
            pulse: {
                type: "f",
                value: 1
            },
            nearFade: {
                type: "v2",
                value: new THREE.Vector2(2 * 0.1, 2 * 0.24)
                // value: new THREE.Vector2(2 * r.insideNear,2 * r.path.waypointIndoorRadius)
            },
            color: {
                type: "c",
                value: new THREE.Color(COLORS.newBlue)  // r.reticuleColor)
            }
        },
        vertexShader: o + "varying vec2 vUv;\nvarying vec4 vPointView;\n\nvoid main() {\n\n  vUv = uv;\n  vPointView = modelViewMatrix * vec4( position, 1.0 );\n  gl_Position = projectionMatrix * vPointView;\n\n}\n",
        fragmentShader: a + "uniform sampler2D map;\nuniform float opacity;\nuniform float pulse; // another opacity, with a different clock\nuniform vec2 nearFade;\nvarying vec2 vUv;\nvarying vec4 vPointView;\nuniform vec3 color;\n\nvoid main() {\n\t// TODO add scroll-in and pulsing behaviors\n\tfloat depthFade = min(1.0, (abs(vPointView.z)-nearFade.x)/(nearFade.y-nearFade.x));\n  vec4 colorFromTexture = texture2D( map, vUv );\t\t// we only use the alpha!\n  gl_FragColor = vec4(color.rgb,\n  \t\t\t\t\t\t(pulse*opacity*colorFromTexture.a * depthFade));\n}\n"
    },
    modelDebug: {
        uniforms: {
            map: {
                type: "t",
                value: null
            },
            modelAlpha: {
                type: "f",
                value: 1 // r.modelAlpha
            },
            depthmapRatio: {
                type: "f",
                value: 0
            },
            opacity: {
                type: "f",
                value: 1
            },
            progress: {
                type: "f",
                value: 0
            },
            considerOcclusion: {
                type: "i",
                value: !1 // r.fancierTransition
            },
            highlightPanoSelection: {
                type: "i",
                value: 0
            },
            useThirdPano: {
                type: "i",
                value: null // r.useThirdPano
            },
            pano0Map: {
                type: "t",
                value: null
            },
            pano0Depth: {
                type: "t",
                value: null
            },
            pano0Position: {
                type: "v3",
                value: new THREE.Vector3
            },
            pano0Matrix: {
                type: "m4",
                value: new THREE.Matrix4
            },
            pano0Weight: {
                type: "f",
                value: null // r.transition.pano0Weight
            },
            pano1Map: {
                type: "t",
                value: null
            },
            pano1Depth: {
                type: "t",
                value: null
            },
            pano1Position: {
                type: "v3",
                value: new THREE.Vector3
            },
            pano1Matrix: {
                type: "m4",
                value: new THREE.Matrix4
            },
            pano1Weight: {
                type: "f",
                value: null // r.transition.pano1Weight
            },
            pano2Map: {
                type: "t",
                value: null
            },
            pano2Depth: {
                type: "t",
                value: null
            },
            pano2Position: {
                type: "v3",
                value: new THREE.Vector3
            },
            pano2Matrix: {
                type: "m4",
                value: new THREE.Matrix4
            },
            pano2Weight: {
                type: "f",
                value: null // r.transition.pano2Weight
            }
        },
        vertexShader: o + "uniform vec3 pano0Position;\nuniform mat4 pano0Matrix;\n\nuniform vec3 pano1Position;\nuniform mat4 pano1Matrix;\n\nuniform vec3 pano2Position;\nuniform mat4 pano2Matrix;\n\nvarying vec2 vUv;\nvarying vec3 vWorldPosition0;\nvarying vec3 vWorldPosition1;\nvarying vec3 vWorldPosition2;\n\nvarying vec4 worldPosition;\n\nvoid main() {\n\n  vUv = uv;\n  worldPosition = modelMatrix * vec4(position, 1.0);\n\n  vec3 positionLocalToPanoCenter0 = worldPosition.xyz - pano0Position;\n  vWorldPosition0 = (vec4(positionLocalToPanoCenter0, 1.0) * pano0Matrix).xyz;\n  vWorldPosition0.x *= -1.0;\n\n  vec3 positionLocalToPanoCenter1 = worldPosition.xyz - pano1Position;\n  vWorldPosition1 = (vec4(positionLocalToPanoCenter1, 1.0) * pano1Matrix).xyz;\n  vWorldPosition1.x *= -1.0;\n\n  vec3 positionLocalToPanoCenter2 = worldPosition.xyz - pano2Position;\n  vWorldPosition2 = (vec4(positionLocalToPanoCenter2, 2.0) * pano2Matrix).xyz;\n  vWorldPosition2.x *= -1.0;\n\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n}\n",
        fragmentShader: a + "uniform sampler2D map;\nuniform float depthmapRatio;\nuniform float modelAlpha;\nuniform float opacity;\nuniform float progress;\nuniform int considerOcclusion;\nuniform int highlightPanoSelection;\nuniform int useThirdPano;\n\nuniform vec3 pano0Position;\nuniform samplerCube pano0Map;\nuniform samplerCube pano0Depth;\nuniform float pano0Weight;\n\nuniform vec3 pano1Position;\nuniform samplerCube pano1Map;\nuniform samplerCube pano1Depth;\nuniform float pano1Weight;\n\nuniform vec3 pano2Position;\nuniform samplerCube pano2Map;\nuniform samplerCube pano2Depth;\nuniform float pano2Weight;\n\nvarying vec2 vUv;\nvarying vec3 vWorldPosition0;\nvarying vec3 vWorldPosition1;\nvarying vec3 vWorldPosition2;\n\nvarying vec4 worldPosition;\n\nvoid main() {\n\n  vec4 depthFromPano0 = textureCube( pano0Depth, vWorldPosition0.xyz );\n  vec4 depthFromPano1 = textureCube( pano1Depth, vWorldPosition1.xyz );\n  vec4 depthFromPano2 = textureCube( pano2Depth, vWorldPosition2.xyz );\n\n  vec4 colorFromPano0 = textureCube( pano0Map, vWorldPosition0.xyz );\n  vec4 colorFromPano1 = textureCube( pano1Map, vWorldPosition1.xyz );\n  vec4 colorFromPano2 = textureCube( pano2Map, vWorldPosition2.xyz );\n\n  float distanceToPano0 = distance(worldPosition.xyz, pano0Position);\n  float distanceToPano1 = distance(worldPosition.xyz, pano1Position);\n  float distanceToPano2 = distance(worldPosition.xyz, pano2Position);\n\n  float cameraToPano0 = distance(cameraPosition.xyz, pano0Position);\n  float cameraToPano1 = distance(cameraPosition.xyz, pano1Position);\n  float cameraToPano2 = distance(cameraPosition.xyz, pano2Position);\n\n  float contributionFromPano0 = cameraToPano0 == 0.0 ? 1000.0 : pano0Weight / cameraToPano0;\n  float contributionFromPano1 = cameraToPano1 == 0.0 ? 1000.0 : pano1Weight / cameraToPano1;\n  float contributionFromPano2 = cameraToPano2 == 0.0 ? 1000.0 : pano2Weight / cameraToPano2;\n\n  contributionFromPano0 *= 1.0 / distanceToPano0;\n  contributionFromPano1 *= 1.0 / distanceToPano1;\n  contributionFromPano2 *= 1.0 / distanceToPano2;\n\n  if(considerOcclusion == 1) {\n    bool occludedFromPano0 = distanceToPano0 / 10.0 > 1.01 - depthFromPano0.x;\n    bool occludedFromPano1 = distanceToPano1 / 10.0 > 1.01 - depthFromPano1.x;\n    bool occludedFromPano2 = distanceToPano2 / 10.0 > 1.01 - depthFromPano2.x;\n\n    if(occludedFromPano0){contributionFromPano0 *= 0.1;}\n    if(occludedFromPano1){contributionFromPano1 *= 0.1;}\n    if(occludedFromPano2){contributionFromPano2 *= 0.1;}\n    //if(occludedFromPano0 && occludedFromPano1 && !occludedFromPano2) { contributionFromPano2 += 0.5; }\n  }\n\n  float contributionSum = contributionFromPano0 + contributionFromPano1 + contributionFromPano2;\n  contributionFromPano0 /= contributionSum;\n  contributionFromPano1 /= contributionSum;\n  contributionFromPano2 /= contributionSum;\n\n  vec4 colorFromPanos = colorFromPano0 * contributionFromPano0;\n  colorFromPanos += colorFromPano1 * contributionFromPano1;\n  colorFromPanos += colorFromPano2 * contributionFromPano2;\n\n  vec4 depthFromPanos = depthFromPano0 * contributionFromPano0;\n  depthFromPanos += depthFromPano1 * contributionFromPano1;\n  depthFromPanos += depthFromPano2 * contributionFromPano2;\n\n  vec4 colorFromTexture = texture2D( map, vUv );\n  colorFromPanos = mix(colorFromPanos, colorFromTexture, modelAlpha);\n\n  if(highlightPanoSelection == 1) {\n    colorFromPanos.r = contributionFromPano0;\n    colorFromPanos.g = contributionFromPano1;\n    colorFromPanos.b = contributionFromPano2;\n  }\n\n  gl_FragColor = vec4(mix(colorFromPanos, depthFromPanos, depthmapRatio).rgb, opacity);\n\n}\n"
    },
    customDepth: {
        uniforms: {
            panoPosition: {
                type: "v3",
                value: new THREE.Vector3
            }
        },
        vertexShader: o + "varying vec4 worldPosition;\n\nvoid main() {\n\n  worldPosition = modelMatrix * vec4(position, 1.0);\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n}\n",
        fragmentShader: a + "uniform vec3 panoPosition;\nvarying vec4 worldPosition;\n\nvoid main() {\n\n  float depth = distance(worldPosition.xyz, panoPosition);\n  float color = 1.0 - depth / 10.0;\n  gl_FragColor = vec4(color, color, color, 1.0);\n\n}\n"
    },
    skysphere: {
        uniforms: {
            radius: {
                type: "f",
                value: 0
            }
        },
        vertexShader: o + "varying vec4 worldPosition;\n\nvoid main() {\n\n  worldPosition = modelMatrix * vec4(position, 1.0);\n  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n}\n",
        fragmentShader: a + "varying vec4 worldPosition;\nuniform float radius;\n\nvoid main() {\n\n  vec4 topColor = vec4(0.094, 0.102, 0.11, 1.0);\n  vec4 bottomColor = vec4(0.2, 0.216, 0.235, 1.0);\n  float normalizedHeight = (worldPosition.y + radius) / (radius * 2.0);\n  float ratio = smoothstep(0.0, 0.5, normalizedHeight);\n  gl_FragColor = mix(bottomColor, topColor, ratio);\n\n}\n"
    }
}


var UTILCOMMON = {
    delayOneFrame: function(e) {
        window.setTimeout(e, 1)
    },
    normalizeUrl: function(e) {
        return e.replace("https://", "http://")
    },
    domainFromUrl: function(e) {
        var t = /^([^:]*:\/\/)?(www\.)?([^\/]+)/.exec(e);
        return t ? t[3] : e
    },
    average: function(e, t) {
        if (0 === e.length)
            return null;
        for (var i = 0, n = 0, r = 0; r < e.length; r++) {
            var o = t ? e[r][t] : e[r];
            i += o,
            n++
        }
        return i / n
    },
    countUnique: function(e) {
        for (var t = {}, i = 0; i < e.length; i++)
            t[e[i]] = 1 + (t[e[i]] || 0);
        return Object.keys(t).length
    },
    averageVectors: function(e, t) {
        var i = new THREE.Vector3;
        if (0 === e.length)
            return i;
        for (var r = 0, o = 0; o < e.length; o++) {
            var a = t ? e[o][t] : e[o];
            i.add(a),
            r++
        }
        return i.divideScalar(r)
    },
    equalLists: function(e, t) {
        if (e.length !== t.length)
            return !1;
        for (var i = 0; i < e.length; i++)
            if (e[i] !== t[i])
                return !1;
        return !0
    },
    lowerMedian: function(e, t) {
        if (0 === e.length)
            return null;
        t = t || 2,
        e.sort(function(e, t) {
            return e - t
        });
        var i = Math.floor(e.length / t);
        return e[i]
    },
    stableSort: function(e, t) {
        return e.map(function(e, t) {
            return {
                value: e,
                index: t
            }
        }).sort(function(e, i) {
            var n = t(e.value, i.value);
            return 0 !== n ? n : e.index - i.index
        }).map(function(e) {
            return e.value
        })
    },
    filterAll: function(e, t) {
        return e.filter(function(e) {
            return t.every(function(t) {
                return t(e)
            })
        })
    },
    formatDate: function(e) {
        return [e.getFullYear(), e.getMonth() + 1, e.getDate()].join("-")
    },
    formatDatetime: function(e) {
        return [e.getFullYear(), e.getMonth() + 1, e.getDate(), e.getHours(), e.getMinutes()].join("-")
    },
    randomString: function(e) {
        for (var t = "", i = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", n = 0; n < e; n++)
            t += i.charAt(Math.floor(Math.random() * i.length));
        return t
    },
    nth: function(e) {
        return e %= 10,
        1 === e ? e + "st" : 2 === e ? e + "nd" : 3 === e ? e + "rd" : e + "th"
    },
    extendObject: function(e, t) {
        return Object.keys(t).forEach(function(i) {
            e[i] = t[i]
        }),
        e
    },
    deepExtend: function e(t) {
        t = t || {};
        for (var i = 1; i < arguments.length; i++) {
            var n = arguments[i];
            if (n)
                for (var r in n)
                    n.hasOwnProperty(r) && ("object" == typeof n[r] ? t[r] = e(t[r], n[r]) : t[r] = n[r])
        }
        return t
    },
    inherit: function(e, t) {
        e.prototype = Object.create(t.prototype),
        e.prototype.constructor = e
    },
    extend: function(e, t) {
        for (var i in t.prototype)
            e.prototype[i] = t.prototype[i]
    }
}


UTILCOMMON.extendObject(Viewmode, CoreViewModes)



Math.sign = function(e) {
    return e < 0 ? -1 : 1
} 


var ModelTextureMaterial = function(e) {
    e = e || {},
    THREE.RawShaderMaterial.call(this, UTILCOMMON.extendObject({
        fragmentShader: SHADERS.model.fragmentShader,
        vertexShader: SHADERS.model.vertexShader,
        uniforms: THREE.UniformsUtils.clone(SHADERS.model.uniforms),
        name: "ModelTextureMaterial"
    }, e))
}
ModelTextureMaterial.prototype = Object.create(THREE.RawShaderMaterial.prototype),
ModelTextureMaterial.prototype.constructor = ModelTextureMaterial,
ModelTextureMaterial.prototype.setProjectedPanos = function(e, t, i) {
    i && (this.uniforms.progress.value = 0),
    e.tiled || (e.skybox.loaded || (e.skybox.needsUpdate = !0),
    e.skybox.loaded = !0),
    this.uniforms.pano0Map.value = e.skybox,
    this.uniforms.pano0Position.value.copy(e.position),
    this.uniforms.pano0Matrix.value.copy(e.skyboxMesh.matrixWorld),
    t.tiled || (t.skybox.loaded || (t.skybox.needsUpdate = !0),
    t.skybox.loaded = !0),
    this.uniforms.pano1Map.value = t.skybox,
    this.uniforms.pano1Position.value.copy(t.position),
    this.uniforms.pano1Matrix.value.copy(t.skyboxMesh.matrixWorld)
}



CHUNK = function(e) {
    this.materialInside = new ModelTextureMaterial({
        side: THREE.DoubleSide
    });
    var t = THREE.UniformsUtils.clone(SHADERS.modelOutside.uniforms);
    this.materialOutside = new THREE.RawShaderMaterial({
        fragmentShader: SHADERS.modelOutside.fragmentShader,
        vertexShader: SHADERS.modelOutside.vertexShader,
        uniforms: t,
        side: THREE.FrontSide,
        name: "chunkOut"
    }),
    THREE.Mesh.call(this, e.geometry, this.materialInside),
    this.name = e.name || "",
    this.textureName = e.textureName,
    this.meshUrl = e.meshUrl
}

CHUNK.prototype = Object.create(THREE.Mesh.prototype),
CHUNK.prototype.setTextureMap = function(e) {
    this.materialInside.uniforms.map.value = e,
    this.materialOutside.uniforms.map.value = e
}
CHUNK.prototype.setMode = function(e) {
    var t = (e === Viewmode.DOLLHOUSE || e === Viewmode.FLOORPLAN) ? this.materialOutside : this.materialInside;
    t.side = e === Viewmode.PANORAMA ? THREE.DoubleSide : THREE.FrontSide,
    t.transparent = this.material.transparent,
    t.uniforms.opacity.value = this.material.uniforms.opacity.value,
    this.material = t
}


protoToken = new Buffer("bWVzc2FnZSBiaW5hcnlfbWVzaCB7CglyZXBlYXRlZCBjaHVua19zaW1wbGUgY2h1bmsgPSAxOwoJcmVwZWF0ZWQgY2h1bmtfcXVhbnRpemVkIHF1YW50aXplZF9jaHVuayA9IDI7Cn0KCi8vIERlZmluaXRpb24gb2YgdmVydGljZXM6IDNEIGNvb3JkaW5hdGVzLCBhbmQgMkQgdGV4dHVyZSBjb29yZGluYXRlcy4KbWVzc2FnZSB2ZXJ0aWNlc19zaW1wbGUgewoJcmVwZWF0ZWQgZmxvYXQgeHl6ID0gMSBbcGFja2VkPXRydWVdOyAgLy8geF8wLHlfMCx6XzAsIHhfMSx5XzEsel8xLCAuLi4KCXJlcGVhdGVkIGZsb2F0IHV2ID0gMiBbcGFja2VkPXRydWVdOyAgLy8gdV8wLHZfMCwgdV8xLHZfMSwgLi4uCn0KCi8vIEluZGV4ZXMgb2YgdmVydGljZXMgb2YgZmFjZXMKbWVzc2FnZSBmYWNlc19zaW1wbGUgewoJcmVwZWF0ZWQgdWludDMyIGZhY2VzID0gMSBbcGFja2VkPXRydWVdOyAvLyBpMDAsaTAxLGkwMiwgaTEwLGkxMSxpMTIsIC4uLgp9CgovLyBBIHNpbXBseSBlbmNvZGVkIGNodW5rLgovLyBUT0RPOiBhZGQgY2h1bmsgcHJvcGVyaXRlcyAoc3VjaCBhcyAicmVmbGVjdGl2ZSIpCm1lc3NhZ2UgY2h1bmtfc2ltcGxlIHsKCW9wdGlvbmFsIHZlcnRpY2VzX3NpbXBsZSB2ZXJ0aWNlcyA9IDE7CglvcHRpb25hbCBmYWNlc19zaW1wbGUgZmFjZXMgPSAyOwoJb3B0aW9uYWwgc3RyaW5nIGNodW5rX25hbWUgPSAzOwoJb3B0aW9uYWwgc3RyaW5nIG1hdGVyaWFsX25hbWUgPSA0Owp9CgovLyBRdWFudGl6ZWQgdmVyc2lvbnMgZm9sbG93OgptZXNzYWdlIHZlcnRpY2VzX3F1YW50aXplZCB7CglvcHRpb25hbCBmbG9hdCBxdWFudGl6YXRpb24gPSAxOwoJcmVwZWF0ZWQgZmxvYXQgdHJhbnNsYXRpb24gPSAyOwoJcmVwZWF0ZWQgc2ludDMyIHggPSAzIFtwYWNrZWQ9dHJ1ZV07CglyZXBlYXRlZCBzaW50MzIgeSA9IDQgW3BhY2tlZD10cnVlXTsKCXJlcGVhdGVkIHNpbnQzMiB6ID0gNSBbcGFja2VkPXRydWVdOwp9CgptZXNzYWdlIHV2X3F1YW50aXplZCB7CglvcHRpb25hbCBzdHJpbmcgbmFtZSA9IDE7CglvcHRpb25hbCBmbG9hdCBxdWFudGl6YXRpb24gPSAyOwoJcmVwZWF0ZWQgc2ludDMyIHUgPSAzIFtwYWNrZWQ9dHJ1ZV07CglyZXBlYXRlZCBzaW50MzIgdiA9IDQgW3BhY2tlZD10cnVlXTsKfQoKLy8gSW5kZXhlcyBvZiB2ZXJ0aWNlcyBvZiBmYWNlcwptZXNzYWdlIGZhY2VzX2NvbXByZXNzZWQgewoJcmVwZWF0ZWQgc2ludDMyIGZhY2VzID0gMSBbcGFja2VkPXRydWVdOyAvLyBpMDAsaTAxLGkwMiwgaTEwLGkxMSxpMTIsIC4uLgp9CgptZXNzYWdlIGNodW5rX3F1YW50aXplZCB7CglvcHRpb25hbCBzdHJpbmcgY2h1bmtfbmFtZSA9IDE7CglvcHRpb25hbCBzdHJpbmcgbWF0ZXJpYWxfbmFtZSA9IDI7CglvcHRpb25hbCB2ZXJ0aWNlc19xdWFudGl6ZWQgdmVydGljZXMgPSAzOwoJcmVwZWF0ZWQgdXZfcXVhbnRpemVkIHV2cyA9IDQ7CglvcHRpb25hbCBmYWNlc19zaW1wbGUgZmFjZXMgPSA1Owp9Cg==", "base64");


module.exports = {
    CoreViewModes:          CoreViewModes,
    Viewmode:               Viewmode,
    COLORS:                 COLORS,
    SHADERS:                SHADERS,   
    UTILCOMMON:             UTILCOMMON,
    ModelTextureMaterial:   ModelTextureMaterial,
    CHUNK:                  CHUNK,
    protoToken:             protoToken
}