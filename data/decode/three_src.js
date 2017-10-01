var n = {
    REVISION: "75"
};

THREE = n;

"function" == typeof define && define.amd ? define("three", n) : "undefined" != typeof i && "undefined" != typeof t && (t.exports = n),
void 0 === Number.EPSILON && (Number.EPSILON = Math.pow(2, -52)),
void 0 === Math.sign && (Math.sign = function(e) {
    return e < 0 ? -1 : e > 0 ? 1 : +e
}
),
void 0 === Function.prototype.name && void 0 !== Object.defineProperty && Object.defineProperty(Function.prototype, "name", {
    get: function() {
        return this.toString().match(/^\s*function\s*(\S*)\s*\(/)[1]
    }
}),
void 0 === Object.assign && Object.defineProperty(Object, "assign", {
    writable: !0,
    configurable: !0,
    value: function(e) {
        "use strict";
        if (void 0 === e || null === e)
            throw new TypeError("Cannot convert first argument to object");
        for (var t = Object(e), i = 1, n = arguments.length; i !== n; ++i) {
            var r = arguments[i];
            if (void 0 !== r && null !== r) {
                r = Object(r);
                for (var o = Object.keys(r), a = 0, s = o.length; a !== s; ++a) {
                    var l = o[a]
                      , h = Object.getOwnPropertyDescriptor(r, l);
                    void 0 !== h && h.enumerable && (t[l] = r[l])
                }
            }
        }
        return t
    }
}),
n.MOUSE = {
    LEFT: 0,
    MIDDLE: 1,
    RIGHT: 2
},
n.CullFaceNone = 0,
n.CullFaceBack = 1,
n.CullFaceFront = 2,
n.CullFaceFrontBack = 3,
n.FrontFaceDirectionCW = 0,
n.FrontFaceDirectionCCW = 1,
n.BasicShadowMap = 0,
n.PCFShadowMap = 1,
n.PCFSoftShadowMap = 2,
n.FrontSide = 0,
n.BackSide = 1,
n.DoubleSide = 2,
n.FlatShading = 1,
n.SmoothShading = 2,
n.NoColors = 0,
n.FaceColors = 1,
n.VertexColors = 2,
n.NoBlending = 0,
n.NormalBlending = 1,
n.AdditiveBlending = 2,
n.SubtractiveBlending = 3,
n.MultiplyBlending = 4,
n.CustomBlending = 5,
n.AddEquation = 100,
n.SubtractEquation = 101,
n.ReverseSubtractEquation = 102,
n.MinEquation = 103,
n.MaxEquation = 104,
n.ZeroFactor = 200,
n.OneFactor = 201,
n.SrcColorFactor = 202,
n.OneMinusSrcColorFactor = 203,
n.SrcAlphaFactor = 204,
n.OneMinusSrcAlphaFactor = 205,
n.DstAlphaFactor = 206,
n.OneMinusDstAlphaFactor = 207,
n.DstColorFactor = 208,
n.OneMinusDstColorFactor = 209,
n.SrcAlphaSaturateFactor = 210,
n.NeverDepth = 0,
n.AlwaysDepth = 1,
n.LessDepth = 2,
n.LessEqualDepth = 3,
n.EqualDepth = 4,
n.GreaterEqualDepth = 5,
n.GreaterDepth = 6,
n.NotEqualDepth = 7,
n.MultiplyOperation = 0,
n.MixOperation = 1,
n.AddOperation = 2,
n.NoToneMapping = 0,
n.LinearToneMapping = 1,
n.ReinhardToneMapping = 2,
n.Uncharted2ToneMapping = 3,
n.CineonToneMapping = 4,
n.UVMapping = 300,
n.CubeReflectionMapping = 301,
n.CubeRefractionMapping = 302,
n.EquirectangularReflectionMapping = 303,
n.EquirectangularRefractionMapping = 304,
n.SphericalReflectionMapping = 305,
n.CubeUVReflectionMapping = 306,
n.CubeUVRefractionMapping = 307,
n.RepeatWrapping = 1e3,
n.ClampToEdgeWrapping = 1001,
n.MirroredRepeatWrapping = 1002,
n.NearestFilter = 1003,
n.NearestMipMapNearestFilter = 1004,
n.NearestMipMapLinearFilter = 1005,
n.LinearFilter = 1006,
n.LinearMipMapNearestFilter = 1007,
n.LinearMipMapLinearFilter = 1008,
n.UnsignedByteType = 1009,
n.ByteType = 1010,
n.ShortType = 1011,
n.UnsignedShortType = 1012,
n.IntType = 1013,
n.UnsignedIntType = 1014,
n.FloatType = 1015,
n.HalfFloatType = 1025,
n.UnsignedShort4444Type = 1016,
n.UnsignedShort5551Type = 1017,
n.UnsignedShort565Type = 1018,
n.AlphaFormat = 1019,
n.RGBFormat = 1020,
n.RGBAFormat = 1021,
n.LuminanceFormat = 1022,
n.LuminanceAlphaFormat = 1023,
n.RGBEFormat = n.RGBAFormat,
n.RGB_S3TC_DXT1_Format = 2001,
n.RGBA_S3TC_DXT1_Format = 2002,
n.RGBA_S3TC_DXT3_Format = 2003,
n.RGBA_S3TC_DXT5_Format = 2004,
n.RGB_PVRTC_4BPPV1_Format = 2100,
n.RGB_PVRTC_2BPPV1_Format = 2101,
n.RGBA_PVRTC_4BPPV1_Format = 2102,
n.RGBA_PVRTC_2BPPV1_Format = 2103,
n.RGB_ETC1_Format = 2151,
n.LoopOnce = 2200,
n.LoopRepeat = 2201,
n.LoopPingPong = 2202,
n.InterpolateDiscrete = 2300,
n.InterpolateLinear = 2301,
n.InterpolateSmooth = 2302,
n.ZeroCurvatureEnding = 2400,
n.ZeroSlopeEnding = 2401,
n.WrapAroundEnding = 2402,
n.TrianglesDrawMode = 0,
n.TriangleStripDrawMode = 1,
n.TriangleFanDrawMode = 2,
n.LinearEncoding = 3e3,
n.sRGBEncoding = 3001,
n.GammaEncoding = 3007,
n.RGBEEncoding = 3002,
n.LogLuvEncoding = 3003,
n.RGBM7Encoding = 3004,
n.RGBM16Encoding = 3005,
n.RGBDEncoding = 3006,
n.Color = function(e) {
    return 3 === arguments.length ? this.fromArray(arguments) : this.set(e)
}
,
n.Color.prototype = {
    constructor: n.Color,
    r: 1,
    g: 1,
    b: 1,
    set: function(e) {
        return e instanceof n.Color ? this.copy(e) : "number" == typeof e ? this.setHex(e) : "string" == typeof e && this.setStyle(e),
        this
    },
    setScalar: function(e) {
        this.r = e,
        this.g = e,
        this.b = e
    },
    setHex: function(e) {
        return e = Math.floor(e),
        this.r = (e >> 16 & 255) / 255,
        this.g = (e >> 8 & 255) / 255,
        this.b = (255 & e) / 255,
        this
    },
    setRGB: function(e, t, i) {
        return this.r = e,
        this.g = t,
        this.b = i,
        this
    },
    setHSL: function() {
        function e(e, t, i) {
            return i < 0 && (i += 1),
            i > 1 && (i -= 1),
            i < 1 / 6 ? e + 6 * (t - e) * i : i < .5 ? t : i < 2 / 3 ? e + 6 * (t - e) * (2 / 3 - i) : e
        }
        return function(t, i, r) {
            if (t = n.Math.euclideanModulo(t, 1),
            i = n.Math.clamp(i, 0, 1),
            r = n.Math.clamp(r, 0, 1),
            0 === i)
                this.r = this.g = this.b = r;
            else {
                var o = r <= .5 ? r * (1 + i) : r + i - r * i
                  , a = 2 * r - o;
                this.r = e(a, o, t + 1 / 3),
                this.g = e(a, o, t),
                this.b = e(a, o, t - 1 / 3)
            }
            return this
        }
    }(),
    setStyle: function(e) {
        function t(t) {
            void 0 !== t && parseFloat(t) < 1 && console.warn("THREE.Color: Alpha component of " + e + " will be ignored.")
        }
        var i;
        if (i = /^((?:rgb|hsl)a?)\(\s*([^\)]*)\)/.exec(e)) {
            var r, o = i[1], a = i[2];
            switch (o) {
            case "rgb":
            case "rgba":
                if (r = /^(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*([0-9]*\.?[0-9]+)\s*)?$/.exec(a))
                    return this.r = Math.min(255, parseInt(r[1], 10)) / 255,
                    this.g = Math.min(255, parseInt(r[2], 10)) / 255,
                    this.b = Math.min(255, parseInt(r[3], 10)) / 255,
                    t(r[5]),
                    this;
                if (r = /^(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(,\s*([0-9]*\.?[0-9]+)\s*)?$/.exec(a))
                    return this.r = Math.min(100, parseInt(r[1], 10)) / 100,
                    this.g = Math.min(100, parseInt(r[2], 10)) / 100,
                    this.b = Math.min(100, parseInt(r[3], 10)) / 100,
                    t(r[5]),
                    this;
                break;
            case "hsl":
            case "hsla":
                if (r = /^([0-9]*\.?[0-9]+)\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(,\s*([0-9]*\.?[0-9]+)\s*)?$/.exec(a)) {
                    var s = parseFloat(r[1]) / 360
                      , l = parseInt(r[2], 10) / 100
                      , h = parseInt(r[3], 10) / 100;
                    return t(r[5]),
                    this.setHSL(s, l, h)
                }
            }
        } else if (i = /^\#([A-Fa-f0-9]+)$/.exec(e)) {
            var c = i[1]
              , u = c.length;
            if (3 === u)
                return this.r = parseInt(c.charAt(0) + c.charAt(0), 16) / 255,
                this.g = parseInt(c.charAt(1) + c.charAt(1), 16) / 255,
                this.b = parseInt(c.charAt(2) + c.charAt(2), 16) / 255,
                this;
            if (6 === u)
                return this.r = parseInt(c.charAt(0) + c.charAt(1), 16) / 255,
                this.g = parseInt(c.charAt(2) + c.charAt(3), 16) / 255,
                this.b = parseInt(c.charAt(4) + c.charAt(5), 16) / 255,
                this
        }
        if (e && e.length > 0) {
            var c = n.ColorKeywords[e];
            void 0 !== c ? this.setHex(c) : console.warn("THREE.Color: Unknown color " + e)
        }
        return this
    },
    clone: function() {
        return new this.constructor(this.r,this.g,this.b)
    },
    copy: function(e) {
        return this.r = e.r,
        this.g = e.g,
        this.b = e.b,
        this
    },
    copyGammaToLinear: function(e, t) {
        return void 0 === t && (t = 2),
        this.r = Math.pow(e.r, t),
        this.g = Math.pow(e.g, t),
        this.b = Math.pow(e.b, t),
        this
    },
    copyLinearToGamma: function(e, t) {
        void 0 === t && (t = 2);
        var i = t > 0 ? 1 / t : 1;
        return this.r = Math.pow(e.r, i),
        this.g = Math.pow(e.g, i),
        this.b = Math.pow(e.b, i),
        this
    },
    convertGammaToLinear: function() {
        var e = this.r
          , t = this.g
          , i = this.b;
        return this.r = e * e,
        this.g = t * t,
        this.b = i * i,
        this
    },
    convertLinearToGamma: function() {
        return this.r = Math.sqrt(this.r),
        this.g = Math.sqrt(this.g),
        this.b = Math.sqrt(this.b),
        this
    },
    getHex: function() {
        return 255 * this.r << 16 ^ 255 * this.g << 8 ^ 255 * this.b << 0
    },
    getHexString: function() {
        return ("000000" + this.getHex().toString(16)).slice(-6)
    },
    getHSL: function(e) {
        var t, i, n = e || {
            h: 0,
            s: 0,
            l: 0
        }, r = this.r, o = this.g, a = this.b, s = Math.max(r, o, a), l = Math.min(r, o, a), h = (l + s) / 2;
        if (l === s)
            t = 0,
            i = 0;
        else {
            var c = s - l;
            switch (i = h <= .5 ? c / (s + l) : c / (2 - s - l),
            s) {
            case r:
                t = (o - a) / c + (o < a ? 6 : 0);
                break;
            case o:
                t = (a - r) / c + 2;
                break;
            case a:
                t = (r - o) / c + 4
            }
            t /= 6
        }
        return n.h = t,
        n.s = i,
        n.l = h,
        n
    },
    getStyle: function() {
        return "rgb(" + (255 * this.r | 0) + "," + (255 * this.g | 0) + "," + (255 * this.b | 0) + ")"
    },
    offsetHSL: function(e, t, i) {
        var n = this.getHSL();
        return n.h += e,
        n.s += t,
        n.l += i,
        this.setHSL(n.h, n.s, n.l),
        this
    },
    add: function(e) {
        return this.r += e.r,
        this.g += e.g,
        this.b += e.b,
        this
    },
    addColors: function(e, t) {
        return this.r = e.r + t.r,
        this.g = e.g + t.g,
        this.b = e.b + t.b,
        this
    },
    addScalar: function(e) {
        return this.r += e,
        this.g += e,
        this.b += e,
        this
    },
    multiply: function(e) {
        return this.r *= e.r,
        this.g *= e.g,
        this.b *= e.b,
        this
    },
    multiplyScalar: function(e) {
        return this.r *= e,
        this.g *= e,
        this.b *= e,
        this
    },
    lerp: function(e, t) {
        return this.r += (e.r - this.r) * t,
        this.g += (e.g - this.g) * t,
        this.b += (e.b - this.b) * t,
        this
    },
    equals: function(e) {
        return e.r === this.r && e.g === this.g && e.b === this.b
    },
    fromArray: function(e, t) {
        return void 0 === t && (t = 0),
        this.r = e[t],
        this.g = e[t + 1],
        this.b = e[t + 2],
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this.r,
        e[t + 1] = this.g,
        e[t + 2] = this.b,
        e
    }
},
n.ColorKeywords = {
    aliceblue: 15792383,
    antiquewhite: 16444375,
    aqua: 65535,
    aquamarine: 8388564,
    azure: 15794175,
    beige: 16119260,
    bisque: 16770244,
    black: 0,
    blanchedalmond: 16772045,
    blue: 255,
    blueviolet: 9055202,
    brown: 10824234,
    burlywood: 14596231,
    cadetblue: 6266528,
    chartreuse: 8388352,
    chocolate: 13789470,
    coral: 16744272,
    cornflowerblue: 6591981,
    cornsilk: 16775388,
    crimson: 14423100,
    cyan: 65535,
    darkblue: 139,
    darkcyan: 35723,
    darkgoldenrod: 12092939,
    darkgray: 11119017,
    darkgreen: 25600,
    darkgrey: 11119017,
    darkkhaki: 12433259,
    darkmagenta: 9109643,
    darkolivegreen: 5597999,
    darkorange: 16747520,
    darkorchid: 10040012,
    darkred: 9109504,
    darksalmon: 15308410,
    darkseagreen: 9419919,
    darkslateblue: 4734347,
    darkslategray: 3100495,
    darkslategrey: 3100495,
    darkturquoise: 52945,
    darkviolet: 9699539,
    deeppink: 16716947,
    deepskyblue: 49151,
    dimgray: 6908265,
    dimgrey: 6908265,
    dodgerblue: 2003199,
    firebrick: 11674146,
    floralwhite: 16775920,
    forestgreen: 2263842,
    fuchsia: 16711935,
    gainsboro: 14474460,
    ghostwhite: 16316671,
    gold: 16766720,
    goldenrod: 14329120,
    gray: 8421504,
    green: 32768,
    greenyellow: 11403055,
    grey: 8421504,
    honeydew: 15794160,
    hotpink: 16738740,
    indianred: 13458524,
    indigo: 4915330,
    ivory: 16777200,
    khaki: 15787660,
    lavender: 15132410,
    lavenderblush: 16773365,
    lawngreen: 8190976,
    lemonchiffon: 16775885,
    lightblue: 11393254,
    lightcoral: 15761536,
    lightcyan: 14745599,
    lightgoldenrodyellow: 16448210,
    lightgray: 13882323,
    lightgreen: 9498256,
    lightgrey: 13882323,
    lightpink: 16758465,
    lightsalmon: 16752762,
    lightseagreen: 2142890,
    lightskyblue: 8900346,
    lightslategray: 7833753,
    lightslategrey: 7833753,
    lightsteelblue: 11584734,
    lightyellow: 16777184,
    lime: 65280,
    limegreen: 3329330,
    linen: 16445670,
    magenta: 16711935,
    maroon: 8388608,
    mediumaquamarine: 6737322,
    mediumblue: 205,
    mediumorchid: 12211667,
    mediumpurple: 9662683,
    mediumseagreen: 3978097,
    mediumslateblue: 8087790,
    mediumspringgreen: 64154,
    mediumturquoise: 4772300,
    mediumvioletred: 13047173,
    midnightblue: 1644912,
    mintcream: 16121850,
    mistyrose: 16770273,
    moccasin: 16770229,
    navajowhite: 16768685,
    navy: 128,
    oldlace: 16643558,
    olive: 8421376,
    olivedrab: 7048739,
    orange: 16753920,
    orangered: 16729344,
    orchid: 14315734,
    palegoldenrod: 15657130,
    palegreen: 10025880,
    paleturquoise: 11529966,
    palevioletred: 14381203,
    papayawhip: 16773077,
    peachpuff: 16767673,
    peru: 13468991,
    pink: 16761035,
    plum: 14524637,
    powderblue: 11591910,
    purple: 8388736,
    red: 16711680,
    rosybrown: 12357519,
    royalblue: 4286945,
    saddlebrown: 9127187,
    salmon: 16416882,
    sandybrown: 16032864,
    seagreen: 3050327,
    seashell: 16774638,
    sienna: 10506797,
    silver: 12632256,
    skyblue: 8900331,
    slateblue: 6970061,
    slategray: 7372944,
    slategrey: 7372944,
    snow: 16775930,
    springgreen: 65407,
    steelblue: 4620980,
    tan: 13808780,
    teal: 32896,
    thistle: 14204888,
    tomato: 16737095,
    turquoise: 4251856,
    violet: 15631086,
    wheat: 16113331,
    white: 16777215,
    whitesmoke: 16119285,
    yellow: 16776960,
    yellowgreen: 10145074
},
n.Quaternion = function(e, t, i, n) {
    this._x = e || 0,
    this._y = t || 0,
    this._z = i || 0,
    this._w = void 0 !== n ? n : 1
}
,
n.Quaternion.prototype = {
    constructor: n.Quaternion,
    get x() {
        return this._x
    },
    set x(e) {
        this._x = e,
        this.onChangeCallback()
    },
    get y() {
        return this._y
    },
    set y(e) {
        this._y = e,
        this.onChangeCallback()
    },
    get z() {
        return this._z
    },
    set z(e) {
        this._z = e,
        this.onChangeCallback()
    },
    get w() {
        return this._w
    },
    set w(e) {
        this._w = e,
        this.onChangeCallback()
    },
    set: function(e, t, i, n) {
        return this._x = e,
        this._y = t,
        this._z = i,
        this._w = n,
        this.onChangeCallback(),
        this
    },
    clone: function() {
        return new this.constructor(this._x,this._y,this._z,this._w)
    },
    copy: function(e) {
        return this._x = e.x,
        this._y = e.y,
        this._z = e.z,
        this._w = e.w,
        this.onChangeCallback(),
        this
    },
    setFromEuler: function(e, t) {
        if (e instanceof n.Euler == !1)
            throw new Error("THREE.Quaternion: .setFromEuler() now expects a Euler rotation rather than a Vector3 and order.");
        var i = Math.cos(e._x / 2)
          , r = Math.cos(e._y / 2)
          , o = Math.cos(e._z / 2)
          , a = Math.sin(e._x / 2)
          , s = Math.sin(e._y / 2)
          , l = Math.sin(e._z / 2)
          , h = e.order;
        return "XYZ" === h ? (this._x = a * r * o + i * s * l,
        this._y = i * s * o - a * r * l,
        this._z = i * r * l + a * s * o,
        this._w = i * r * o - a * s * l) : "YXZ" === h ? (this._x = a * r * o + i * s * l,
        this._y = i * s * o - a * r * l,
        this._z = i * r * l - a * s * o,
        this._w = i * r * o + a * s * l) : "ZXY" === h ? (this._x = a * r * o - i * s * l,
        this._y = i * s * o + a * r * l,
        this._z = i * r * l + a * s * o,
        this._w = i * r * o - a * s * l) : "ZYX" === h ? (this._x = a * r * o - i * s * l,
        this._y = i * s * o + a * r * l,
        this._z = i * r * l - a * s * o,
        this._w = i * r * o + a * s * l) : "YZX" === h ? (this._x = a * r * o + i * s * l,
        this._y = i * s * o + a * r * l,
        this._z = i * r * l - a * s * o,
        this._w = i * r * o - a * s * l) : "XZY" === h && (this._x = a * r * o - i * s * l,
        this._y = i * s * o - a * r * l,
        this._z = i * r * l + a * s * o,
        this._w = i * r * o + a * s * l),
        t !== !1 && this.onChangeCallback(),
        this
    },
    setFromAxisAngle: function(e, t) {
        var i = t / 2
          , n = Math.sin(i);
        return this._x = e.x * n,
        this._y = e.y * n,
        this._z = e.z * n,
        this._w = Math.cos(i),
        this.onChangeCallback(),
        this
    },
    setFromRotationMatrix: function(e) {
        var t, i = e.elements, n = i[0], r = i[4], o = i[8], a = i[1], s = i[5], l = i[9], h = i[2], c = i[6], u = i[10], d = n + s + u;
        return d > 0 ? (t = .5 / Math.sqrt(d + 1),
        this._w = .25 / t,
        this._x = (c - l) * t,
        this._y = (o - h) * t,
        this._z = (a - r) * t) : n > s && n > u ? (t = 2 * Math.sqrt(1 + n - s - u),
        this._w = (c - l) / t,
        this._x = .25 * t,
        this._y = (r + a) / t,
        this._z = (o + h) / t) : s > u ? (t = 2 * Math.sqrt(1 + s - n - u),
        this._w = (o - h) / t,
        this._x = (r + a) / t,
        this._y = .25 * t,
        this._z = (l + c) / t) : (t = 2 * Math.sqrt(1 + u - n - s),
        this._w = (a - r) / t,
        this._x = (o + h) / t,
        this._y = (l + c) / t,
        this._z = .25 * t),
        this.onChangeCallback(),
        this
    },
    setFromUnitVectors: function() {
        var e, t, i = 1e-6;
        return function(r, o) {
            return void 0 === e && (e = new n.Vector3),
            t = r.dot(o) + 1,
            t < i ? (t = 0,
            Math.abs(r.x) > Math.abs(r.z) ? e.set(-r.y, r.x, 0) : e.set(0, -r.z, r.y)) : e.crossVectors(r, o),
            this._x = e.x,
            this._y = e.y,
            this._z = e.z,
            this._w = t,
            this.normalize(),
            this
        }
    }(),
    inverse: function() {
        return this.conjugate().normalize(),
        this
    },
    conjugate: function() {
        return this._x *= -1,
        this._y *= -1,
        this._z *= -1,
        this.onChangeCallback(),
        this
    },
    dot: function(e) {
        return this._x * e._x + this._y * e._y + this._z * e._z + this._w * e._w
    },
    lengthSq: function() {
        return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w
    },
    length: function() {
        return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w)
    },
    normalize: function() {
        var e = this.length();
        return 0 === e ? (this._x = 0,
        this._y = 0,
        this._z = 0,
        this._w = 1) : (e = 1 / e,
        this._x = this._x * e,
        this._y = this._y * e,
        this._z = this._z * e,
        this._w = this._w * e),
        this.onChangeCallback(),
        this
    },
    multiply: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Quaternion: .multiply() now only accepts one argument. Use .multiplyQuaternions( a, b ) instead."),
        this.multiplyQuaternions(e, t)) : this.multiplyQuaternions(this, e)
    },
    multiplyQuaternions: function(e, t) {
        var i = e._x
          , n = e._y
          , r = e._z
          , o = e._w
          , a = t._x
          , s = t._y
          , l = t._z
          , h = t._w;
        return this._x = i * h + o * a + n * l - r * s,
        this._y = n * h + o * s + r * a - i * l,
        this._z = r * h + o * l + i * s - n * a,
        this._w = o * h - i * a - n * s - r * l,
        this.onChangeCallback(),
        this
    },
    slerp: function(e, t) {
        if (0 === t)
            return this;
        if (1 === t)
            return this.copy(e);
        var i = this._x
          , n = this._y
          , r = this._z
          , o = this._w
          , a = o * e._w + i * e._x + n * e._y + r * e._z;
        if (a < 0 ? (this._w = -e._w,
        this._x = -e._x,
        this._y = -e._y,
        this._z = -e._z,
        a = -a) : this.copy(e),
        a >= 1)
            return this._w = o,
            this._x = i,
            this._y = n,
            this._z = r,
            this;
        var s = Math.sqrt(1 - a * a);
        if (Math.abs(s) < .001)
            return this._w = .5 * (o + this._w),
            this._x = .5 * (i + this._x),
            this._y = .5 * (n + this._y),
            this._z = .5 * (r + this._z),
            this;
        var l = Math.atan2(s, a)
          , h = Math.sin((1 - t) * l) / s
          , c = Math.sin(t * l) / s;
        return this._w = o * h + this._w * c,
        this._x = i * h + this._x * c,
        this._y = n * h + this._y * c,
        this._z = r * h + this._z * c,
        this.onChangeCallback(),
        this
    },
    equals: function(e) {
        return e._x === this._x && e._y === this._y && e._z === this._z && e._w === this._w
    },
    fromArray: function(e, t) {
        return void 0 === t && (t = 0),
        this._x = e[t],
        this._y = e[t + 1],
        this._z = e[t + 2],
        this._w = e[t + 3],
        this.onChangeCallback(),
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this._x,
        e[t + 1] = this._y,
        e[t + 2] = this._z,
        e[t + 3] = this._w,
        e
    },
    onChange: function(e) {
        return this.onChangeCallback = e,
        this
    },
    onChangeCallback: function() {}
},
Object.assign(n.Quaternion, {
    slerp: function(e, t, i, n) {
        return i.copy(e).slerp(t, n)
    },
    slerpFlat: function(e, t, i, n, r, o, a) {
        var s = i[n + 0]
          , l = i[n + 1]
          , h = i[n + 2]
          , c = i[n + 3]
          , u = r[o + 0]
          , d = r[o + 1]
          , p = r[o + 2]
          , f = r[o + 3];
        if (c !== f || s !== u || l !== d || h !== p) {
            var g = 1 - a
              , m = s * u + l * d + h * p + c * f
              , v = m >= 0 ? 1 : -1
              , A = 1 - m * m;
            if (A > Number.EPSILON) {
                var y = Math.sqrt(A)
                  , C = Math.atan2(y, m * v);
                g = Math.sin(g * C) / y,
                a = Math.sin(a * C) / y
            }
            var I = a * v;
            if (s = s * g + u * I,
            l = l * g + d * I,
            h = h * g + p * I,
            c = c * g + f * I,
            g === 1 - a) {
                var b = 1 / Math.sqrt(s * s + l * l + h * h + c * c);
                s *= b,
                l *= b,
                h *= b,
                c *= b
            }
        }
        e[t] = s,
        e[t + 1] = l,
        e[t + 2] = h,
        e[t + 3] = c
    }
}),
n.Vector2 = function(e, t) {
    this.x = e || 0,
    this.y = t || 0
}
,
n.Vector2.prototype = {
    constructor: n.Vector2,
    get width() {
        return this.x
    },
    set width(e) {
        this.x = e
    },
    get height() {
        return this.y
    },
    set height(e) {
        this.y = e
    },
    set: function(e, t) {
        return this.x = e,
        this.y = t,
        this
    },
    setScalar: function(e) {
        return this.x = e,
        this.y = e,
        this
    },
    setX: function(e) {
        return this.x = e,
        this
    },
    setY: function(e) {
        return this.y = e,
        this
    },
    setComponent: function(e, t) {
        switch (e) {
        case 0:
            this.x = t;
            break;
        case 1:
            this.y = t;
            break;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    getComponent: function(e) {
        switch (e) {
        case 0:
            return this.x;
        case 1:
            return this.y;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    clone: function() {
        return new this.constructor(this.x,this.y)
    },
    copy: function(e) {
        return this.x = e.x,
        this.y = e.y,
        this
    },
    add: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector2: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),
        this.addVectors(e, t)) : (this.x += e.x,
        this.y += e.y,
        this)
    },
    addScalar: function(e) {
        return this.x += e,
        this.y += e,
        this
    },
    addVectors: function(e, t) {
        return this.x = e.x + t.x,
        this.y = e.y + t.y,
        this
    },
    addScaledVector: function(e, t) {
        return this.x += e.x * t,
        this.y += e.y * t,
        this
    },
    sub: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector2: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),
        this.subVectors(e, t)) : (this.x -= e.x,
        this.y -= e.y,
        this)
    },
    subScalar: function(e) {
        return this.x -= e,
        this.y -= e,
        this
    },
    subVectors: function(e, t) {
        return this.x = e.x - t.x,
        this.y = e.y - t.y,
        this
    },
    multiply: function(e) {
        return this.x *= e.x,
        this.y *= e.y,
        this
    },
    multiplyScalar: function(e) {
        return isFinite(e) ? (this.x *= e,
        this.y *= e) : (this.x = 0,
        this.y = 0),
        this
    },
    divide: function(e) {
        return this.x /= e.x,
        this.y /= e.y,
        this
    },
    divideScalar: function(e) {
        return this.multiplyScalar(1 / e)
    },
    min: function(e) {
        return this.x = Math.min(this.x, e.x),
        this.y = Math.min(this.y, e.y),
        this
    },
    max: function(e) {
        return this.x = Math.max(this.x, e.x),
        this.y = Math.max(this.y, e.y),
        this
    },
    clamp: function(e, t) {
        return this.x = Math.max(e.x, Math.min(t.x, this.x)),
        this.y = Math.max(e.y, Math.min(t.y, this.y)),
        this
    },
    clampScalar: function() {
        var e, t;
        return function(i, r) {
            return void 0 === e && (e = new n.Vector2,
            t = new n.Vector2),
            e.set(i, i),
            t.set(r, r),
            this.clamp(e, t)
        }
    }(),
    clampLength: function(e, t) {
        var i = this.length();
        return this.multiplyScalar(Math.max(e, Math.min(t, i)) / i),
        this
    },
    floor: function() {
        return this.x = Math.floor(this.x),
        this.y = Math.floor(this.y),
        this
    },
    ceil: function() {
        return this.x = Math.ceil(this.x),
        this.y = Math.ceil(this.y),
        this
    },
    round: function() {
        return this.x = Math.round(this.x),
        this.y = Math.round(this.y),
        this
    },
    roundToZero: function() {
        return this.x = this.x < 0 ? Math.ceil(this.x) : Math.floor(this.x),
        this.y = this.y < 0 ? Math.ceil(this.y) : Math.floor(this.y),
        this
    },
    negate: function() {
        return this.x = -this.x,
        this.y = -this.y,
        this
    },
    dot: function(e) {
        return this.x * e.x + this.y * e.y
    },
    lengthSq: function() {
        return this.x * this.x + this.y * this.y
    },
    length: function() {
        return Math.sqrt(this.x * this.x + this.y * this.y)
    },
    lengthManhattan: function() {
        return Math.abs(this.x) + Math.abs(this.y)
    },
    normalize: function() {
        return this.divideScalar(this.length())
    },
    angle: function() {
        var e = Math.atan2(this.y, this.x);
        return e < 0 && (e += 2 * Math.PI),
        e
    },
    distanceTo: function(e) {
        return Math.sqrt(this.distanceToSquared(e))
    },
    distanceToSquared: function(e) {
        var t = this.x - e.x
          , i = this.y - e.y;
        return t * t + i * i
    },
    setLength: function(e) {
        return this.multiplyScalar(e / this.length())
    },
    lerp: function(e, t) {
        return this.x += (e.x - this.x) * t,
        this.y += (e.y - this.y) * t,
        this
    },
    lerpVectors: function(e, t, i) {
        return this.subVectors(t, e).multiplyScalar(i).add(e),
        this
    },
    equals: function(e) {
        return e.x === this.x && e.y === this.y
    },
    fromArray: function(e, t) {
        return void 0 === t && (t = 0),
        this.x = e[t],
        this.y = e[t + 1],
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this.x,
        e[t + 1] = this.y,
        e
    },
    fromAttribute: function(e, t, i) {
        return void 0 === i && (i = 0),
        t = t * e.itemSize + i,
        this.x = e.array[t],
        this.y = e.array[t + 1],
        this
    },
    rotateAround: function(e, t) {
        var i = Math.cos(t)
          , n = Math.sin(t)
          , r = this.x - e.x
          , o = this.y - e.y;
        return this.x = r * i - o * n + e.x,
        this.y = r * n + o * i + e.y,
        this
    }
},
n.Vector3 = function(e, t, i) {
    this.x = e || 0,
    this.y = t || 0,
    this.z = i || 0
}
,
n.Vector3.prototype = {
    constructor: n.Vector3,
    set: function(e, t, i) {
        return this.x = e,
        this.y = t,
        this.z = i,
        this
    },
    setScalar: function(e) {
        return this.x = e,
        this.y = e,
        this.z = e,
        this
    },
    setX: function(e) {
        return this.x = e,
        this
    },
    setY: function(e) {
        return this.y = e,
        this
    },
    setZ: function(e) {
        return this.z = e,
        this
    },
    setComponent: function(e, t) {
        switch (e) {
        case 0:
            this.x = t;
            break;
        case 1:
            this.y = t;
            break;
        case 2:
            this.z = t;
            break;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    getComponent: function(e) {
        switch (e) {
        case 0:
            return this.x;
        case 1:
            return this.y;
        case 2:
            return this.z;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    clone: function() {
        return new this.constructor(this.x,this.y,this.z)
    },
    copy: function(e) {
        return this.x = e.x,
        this.y = e.y,
        this.z = e.z,
        this
    },
    add: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector3: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),
        this.addVectors(e, t)) : (this.x += e.x,
        this.y += e.y,
        this.z += e.z,
        this)
    },
    addScalar: function(e) {
        return this.x += e,
        this.y += e,
        this.z += e,
        this
    },
    addVectors: function(e, t) {
        return this.x = e.x + t.x,
        this.y = e.y + t.y,
        this.z = e.z + t.z,
        this
    },
    addScaledVector: function(e, t) {
        return this.x += e.x * t,
        this.y += e.y * t,
        this.z += e.z * t,
        this
    },
    sub: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector3: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),
        this.subVectors(e, t)) : (this.x -= e.x,
        this.y -= e.y,
        this.z -= e.z,
        this)
    },
    subScalar: function(e) {
        return this.x -= e,
        this.y -= e,
        this.z -= e,
        this
    },
    subVectors: function(e, t) {
        return this.x = e.x - t.x,
        this.y = e.y - t.y,
        this.z = e.z - t.z,
        this
    },
    multiply: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector3: .multiply() now only accepts one argument. Use .multiplyVectors( a, b ) instead."),
        this.multiplyVectors(e, t)) : (this.x *= e.x,
        this.y *= e.y,
        this.z *= e.z,
        this)
    },
    multiplyScalar: function(e) {
        return isFinite(e) ? (this.x *= e,
        this.y *= e,
        this.z *= e) : (this.x = 0,
        this.y = 0,
        this.z = 0),
        this
    },
    multiplyVectors: function(e, t) {
        return this.x = e.x * t.x,
        this.y = e.y * t.y,
        this.z = e.z * t.z,
        this
    },
    applyEuler: function() {
        var e;
        return function(t) {
            return t instanceof n.Euler == !1 && console.error("THREE.Vector3: .applyEuler() now expects an Euler rotation rather than a Vector3 and order."),
            void 0 === e && (e = new n.Quaternion),
            this.applyQuaternion(e.setFromEuler(t)),
            this
        }
    }(),
    applyAxisAngle: function() {
        var e;
        return function(t, i) {
            return void 0 === e && (e = new n.Quaternion),
            this.applyQuaternion(e.setFromAxisAngle(t, i)),
            this
        }
    }(),
    applyMatrix3: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = e.elements;
        return this.x = r[0] * t + r[3] * i + r[6] * n,
        this.y = r[1] * t + r[4] * i + r[7] * n,
        this.z = r[2] * t + r[5] * i + r[8] * n,
        this
    },
    applyMatrix4: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = e.elements;
        return this.x = r[0] * t + r[4] * i + r[8] * n + r[12],
        this.y = r[1] * t + r[5] * i + r[9] * n + r[13],
        this.z = r[2] * t + r[6] * i + r[10] * n + r[14],
        this
    },
    applyProjection: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = e.elements
          , o = 1 / (r[3] * t + r[7] * i + r[11] * n + r[15]);
        return this.x = (r[0] * t + r[4] * i + r[8] * n + r[12]) * o,
        this.y = (r[1] * t + r[5] * i + r[9] * n + r[13]) * o,
        this.z = (r[2] * t + r[6] * i + r[10] * n + r[14]) * o,
        this
    },
    applyQuaternion: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = e.x
          , o = e.y
          , a = e.z
          , s = e.w
          , l = s * t + o * n - a * i
          , h = s * i + a * t - r * n
          , c = s * n + r * i - o * t
          , u = -r * t - o * i - a * n;
        return this.x = l * s + u * -r + h * -a - c * -o,
        this.y = h * s + u * -o + c * -r - l * -a,
        this.z = c * s + u * -a + l * -o - h * -r,
        this
    },
    project: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.multiplyMatrices(t.projectionMatrix, e.getInverse(t.matrixWorld)),
            this.applyProjection(e)
        }
    }(),
    unproject: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.multiplyMatrices(t.matrixWorld, e.getInverse(t.projectionMatrix)),
            this.applyProjection(e)
        }
    }(),
    transformDirection: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = e.elements;
        return this.x = r[0] * t + r[4] * i + r[8] * n,
        this.y = r[1] * t + r[5] * i + r[9] * n,
        this.z = r[2] * t + r[6] * i + r[10] * n,
        this.normalize(),
        this
    },
    divide: function(e) {
        return this.x /= e.x,
        this.y /= e.y,
        this.z /= e.z,
        this
    },
    divideScalar: function(e) {
        return this.multiplyScalar(1 / e)
    },
    min: function(e) {
        return this.x = Math.min(this.x, e.x),
        this.y = Math.min(this.y, e.y),
        this.z = Math.min(this.z, e.z),
        this
    },
    max: function(e) {
        return this.x = Math.max(this.x, e.x),
        this.y = Math.max(this.y, e.y),
        this.z = Math.max(this.z, e.z),
        this
    },
    clamp: function(e, t) {
        return this.x = Math.max(e.x, Math.min(t.x, this.x)),
        this.y = Math.max(e.y, Math.min(t.y, this.y)),
        this.z = Math.max(e.z, Math.min(t.z, this.z)),
        this
    },
    clampScalar: function() {
        var e, t;
        return function(i, r) {
            return void 0 === e && (e = new n.Vector3,
            t = new n.Vector3),
            e.set(i, i, i),
            t.set(r, r, r),
            this.clamp(e, t)
        }
    }(),
    clampLength: function(e, t) {
        var i = this.length();
        return this.multiplyScalar(Math.max(e, Math.min(t, i)) / i),
        this
    },
    floor: function() {
        return this.x = Math.floor(this.x),
        this.y = Math.floor(this.y),
        this.z = Math.floor(this.z),
        this
    },
    ceil: function() {
        return this.x = Math.ceil(this.x),
        this.y = Math.ceil(this.y),
        this.z = Math.ceil(this.z),
        this
    },
    round: function() {
        return this.x = Math.round(this.x),
        this.y = Math.round(this.y),
        this.z = Math.round(this.z),
        this
    },
    roundToZero: function() {
        return this.x = this.x < 0 ? Math.ceil(this.x) : Math.floor(this.x),
        this.y = this.y < 0 ? Math.ceil(this.y) : Math.floor(this.y),
        this.z = this.z < 0 ? Math.ceil(this.z) : Math.floor(this.z),
        this
    },
    negate: function() {
        return this.x = -this.x,
        this.y = -this.y,
        this.z = -this.z,
        this
    },
    dot: function(e) {
        return this.x * e.x + this.y * e.y + this.z * e.z
    },
    lengthSq: function() {
        return this.x * this.x + this.y * this.y + this.z * this.z
    },
    length: function() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z)
    },
    lengthManhattan: function() {
        return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z)
    },
    normalize: function() {
        return this.divideScalar(this.length())
    },
    setLength: function(e) {
        return this.multiplyScalar(e / this.length())
    },
    lerp: function(e, t) {
        return this.x += (e.x - this.x) * t,
        this.y += (e.y - this.y) * t,
        this.z += (e.z - this.z) * t,
        this
    },
    lerpVectors: function(e, t, i) {
        return this.subVectors(t, e).multiplyScalar(i).add(e),
        this
    },
    cross: function(e, t) {
        if (void 0 !== t)
            return console.warn("THREE.Vector3: .cross() now only accepts one argument. Use .crossVectors( a, b ) instead."),
            this.crossVectors(e, t);
        var i = this.x
          , n = this.y
          , r = this.z;
        return this.x = n * e.z - r * e.y,
        this.y = r * e.x - i * e.z,
        this.z = i * e.y - n * e.x,
        this
    },
    crossVectors: function(e, t) {
        var i = e.x
          , n = e.y
          , r = e.z
          , o = t.x
          , a = t.y
          , s = t.z;
        return this.x = n * s - r * a,
        this.y = r * o - i * s,
        this.z = i * a - n * o,
        this
    },
    projectOnVector: function() {
        var e, t;
        return function(i) {
            return void 0 === e && (e = new n.Vector3),
            e.copy(i).normalize(),
            t = this.dot(e),
            this.copy(e).multiplyScalar(t)
        }
    }(),
    projectOnPlane: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Vector3),
            e.copy(this).projectOnVector(t),
            this.sub(e)
        }
    }(),
    reflect: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Vector3),
            this.sub(e.copy(t).multiplyScalar(2 * this.dot(t)))
        }
    }(),
    angleTo: function(e) {
        var t = this.dot(e) / Math.sqrt(this.lengthSq() * e.lengthSq());
        return Math.acos(n.Math.clamp(t, -1, 1))
    },
    distanceTo: function(e) {
        return Math.sqrt(this.distanceToSquared(e))
    },
    distanceToSquared: function(e) {
        var t = this.x - e.x
          , i = this.y - e.y
          , n = this.z - e.z;
        return t * t + i * i + n * n
    },
    setFromSpherical: function(e) {
        var t = Math.sin(e.phi) * e.radius;
        return this.x = t * Math.sin(e.theta),
        this.y = Math.cos(e.phi) * e.radius,
        this.z = t * Math.cos(e.theta),
        this
    },
    setFromMatrixPosition: function(e) {
        return this.setFromMatrixColumn(e, 3)
    },
    setFromMatrixScale: function(e) {
        var t = this.setFromMatrixColumn(e, 0).length()
          , i = this.setFromMatrixColumn(e, 1).length()
          , n = this.setFromMatrixColumn(e, 2).length();
        return this.x = t,
        this.y = i,
        this.z = n,
        this
    },
    setFromMatrixColumn: function(e, t) {
        return "number" == typeof e && (console.warn("THREE.Vector3: setFromMatrixColumn now expects ( matrix, index )."),
        e = arguments[1],
        t = arguments[0]),
        this.fromArray(e.elements, 4 * t)
    },
    equals: function(e) {
        return e.x === this.x && e.y === this.y && e.z === this.z
    },
    fromArray: function(e, t) {
        return void 0 === t && (t = 0),
        this.x = e[t],
        this.y = e[t + 1],
        this.z = e[t + 2],
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this.x,
        e[t + 1] = this.y,
        e[t + 2] = this.z,
        e
    },
    fromAttribute: function(e, t, i) {
        return void 0 === i && (i = 0),
        t = t * e.itemSize + i,
        this.x = e.array[t],
        this.y = e.array[t + 1],
        this.z = e.array[t + 2],
        this
    }
},
n.Vector4 = function(e, t, i, n) {
    this.x = e || 0,
    this.y = t || 0,
    this.z = i || 0,
    this.w = void 0 !== n ? n : 1
}
,
n.Vector4.prototype = {
    constructor: n.Vector4,
    set: function(e, t, i, n) {
        return this.x = e,
        this.y = t,
        this.z = i,
        this.w = n,
        this
    },
    setScalar: function(e) {
        return this.x = e,
        this.y = e,
        this.z = e,
        this.w = e,
        this
    },
    setX: function(e) {
        return this.x = e,
        this
    },
    setY: function(e) {
        return this.y = e,
        this
    },
    setZ: function(e) {
        return this.z = e,
        this
    },
    setW: function(e) {
        return this.w = e,
        this
    },
    setComponent: function(e, t) {
        switch (e) {
        case 0:
            this.x = t;
            break;
        case 1:
            this.y = t;
            break;
        case 2:
            this.z = t;
            break;
        case 3:
            this.w = t;
            break;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    getComponent: function(e) {
        switch (e) {
        case 0:
            return this.x;
        case 1:
            return this.y;
        case 2:
            return this.z;
        case 3:
            return this.w;
        default:
            throw new Error("index is out of range: " + e)
        }
    },
    clone: function() {
        return new this.constructor(this.x,this.y,this.z,this.w)
    },
    copy: function(e) {
        return this.x = e.x,
        this.y = e.y,
        this.z = e.z,
        this.w = void 0 !== e.w ? e.w : 1,
        this
    },
    add: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector4: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),
        this.addVectors(e, t)) : (this.x += e.x,
        this.y += e.y,
        this.z += e.z,
        this.w += e.w,
        this)
    },
    addScalar: function(e) {
        return this.x += e,
        this.y += e,
        this.z += e,
        this.w += e,
        this
    },
    addVectors: function(e, t) {
        return this.x = e.x + t.x,
        this.y = e.y + t.y,
        this.z = e.z + t.z,
        this.w = e.w + t.w,
        this
    },
    addScaledVector: function(e, t) {
        return this.x += e.x * t,
        this.y += e.y * t,
        this.z += e.z * t,
        this.w += e.w * t,
        this
    },
    sub: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Vector4: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),
        this.subVectors(e, t)) : (this.x -= e.x,
        this.y -= e.y,
        this.z -= e.z,
        this.w -= e.w,
        this)
    },
    subScalar: function(e) {
        return this.x -= e,
        this.y -= e,
        this.z -= e,
        this.w -= e,
        this
    },
    subVectors: function(e, t) {
        return this.x = e.x - t.x,
        this.y = e.y - t.y,
        this.z = e.z - t.z,
        this.w = e.w - t.w,
        this
    },
    multiplyScalar: function(e) {
        return isFinite(e) ? (this.x *= e,
        this.y *= e,
        this.z *= e,
        this.w *= e) : (this.x = 0,
        this.y = 0,
        this.z = 0,
        this.w = 0),
        this
    },
    applyMatrix4: function(e) {
        var t = this.x
          , i = this.y
          , n = this.z
          , r = this.w
          , o = e.elements;
        return this.x = o[0] * t + o[4] * i + o[8] * n + o[12] * r,
        this.y = o[1] * t + o[5] * i + o[9] * n + o[13] * r,
        this.z = o[2] * t + o[6] * i + o[10] * n + o[14] * r,
        this.w = o[3] * t + o[7] * i + o[11] * n + o[15] * r,
        this
    },
    divideScalar: function(e) {
        return this.multiplyScalar(1 / e)
    },
    setAxisAngleFromQuaternion: function(e) {
        this.w = 2 * Math.acos(e.w);
        var t = Math.sqrt(1 - e.w * e.w);
        return t < 1e-4 ? (this.x = 1,
        this.y = 0,
        this.z = 0) : (this.x = e.x / t,
        this.y = e.y / t,
        this.z = e.z / t),
        this
    },
    setAxisAngleFromRotationMatrix: function(e) {
        var t, i, n, r, o = .01, a = .1, s = e.elements, l = s[0], h = s[4], c = s[8], u = s[1], d = s[5], p = s[9], f = s[2], g = s[6], m = s[10];
        if (Math.abs(h - u) < o && Math.abs(c - f) < o && Math.abs(p - g) < o) {
            if (Math.abs(h + u) < a && Math.abs(c + f) < a && Math.abs(p + g) < a && Math.abs(l + d + m - 3) < a)
                return this.set(1, 0, 0, 0),
                this;
            t = Math.PI;
            var v = (l + 1) / 2
              , A = (d + 1) / 2
              , y = (m + 1) / 2
              , C = (h + u) / 4
              , I = (c + f) / 4
              , b = (p + g) / 4;
            return v > A && v > y ? v < o ? (i = 0,
            n = .707106781,
            r = .707106781) : (i = Math.sqrt(v),
            n = C / i,
            r = I / i) : A > y ? A < o ? (i = .707106781,
            n = 0,
            r = .707106781) : (n = Math.sqrt(A),
            i = C / n,
            r = b / n) : y < o ? (i = .707106781,
            n = .707106781,
            r = 0) : (r = Math.sqrt(y),
            i = I / r,
            n = b / r),
            this.set(i, n, r, t),
            this
        }
        var w = Math.sqrt((g - p) * (g - p) + (c - f) * (c - f) + (u - h) * (u - h));
        return Math.abs(w) < .001 && (w = 1),
        this.x = (g - p) / w,
        this.y = (c - f) / w,
        this.z = (u - h) / w,
        this.w = Math.acos((l + d + m - 1) / 2),
        this
    },
    min: function(e) {
        return this.x = Math.min(this.x, e.x),
        this.y = Math.min(this.y, e.y),
        this.z = Math.min(this.z, e.z),
        this.w = Math.min(this.w, e.w),
        this
    },
    max: function(e) {
        return this.x = Math.max(this.x, e.x),
        this.y = Math.max(this.y, e.y),
        this.z = Math.max(this.z, e.z),
        this.w = Math.max(this.w, e.w),
        this
    },
    clamp: function(e, t) {
        return this.x = Math.max(e.x, Math.min(t.x, this.x)),
        this.y = Math.max(e.y, Math.min(t.y, this.y)),
        this.z = Math.max(e.z, Math.min(t.z, this.z)),
        this.w = Math.max(e.w, Math.min(t.w, this.w)),
        this
    },
    clampScalar: function() {
        var e, t;
        return function(i, r) {
            return void 0 === e && (e = new n.Vector4,
            t = new n.Vector4),
            e.set(i, i, i, i),
            t.set(r, r, r, r),
            this.clamp(e, t)
        }
    }(),
    floor: function() {
        return this.x = Math.floor(this.x),
        this.y = Math.floor(this.y),
        this.z = Math.floor(this.z),
        this.w = Math.floor(this.w),
        this
    },
    ceil: function() {
        return this.x = Math.ceil(this.x),
        this.y = Math.ceil(this.y),
        this.z = Math.ceil(this.z),
        this.w = Math.ceil(this.w),
        this
    },
    round: function() {
        return this.x = Math.round(this.x),
        this.y = Math.round(this.y),
        this.z = Math.round(this.z),
        this.w = Math.round(this.w),
        this
    },
    roundToZero: function() {
        return this.x = this.x < 0 ? Math.ceil(this.x) : Math.floor(this.x),
        this.y = this.y < 0 ? Math.ceil(this.y) : Math.floor(this.y),
        this.z = this.z < 0 ? Math.ceil(this.z) : Math.floor(this.z),
        this.w = this.w < 0 ? Math.ceil(this.w) : Math.floor(this.w),
        this
    },
    negate: function() {
        return this.x = -this.x,
        this.y = -this.y,
        this.z = -this.z,
        this.w = -this.w,
        this
    },
    dot: function(e) {
        return this.x * e.x + this.y * e.y + this.z * e.z + this.w * e.w
    },
    lengthSq: function() {
        return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w
    },
    length: function() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)
    },
    lengthManhattan: function() {
        return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w)
    },
    normalize: function() {
        return this.divideScalar(this.length())
    },
    setLength: function(e) {
        return this.multiplyScalar(e / this.length())
    },
    lerp: function(e, t) {
        return this.x += (e.x - this.x) * t,
        this.y += (e.y - this.y) * t,
        this.z += (e.z - this.z) * t,
        this.w += (e.w - this.w) * t,
        this
    },
    lerpVectors: function(e, t, i) {
        return this.subVectors(t, e).multiplyScalar(i).add(e),
        this
    },
    equals: function(e) {
        return e.x === this.x && e.y === this.y && e.z === this.z && e.w === this.w
    },
    fromArray: function(e, t) {
        return void 0 === t && (t = 0),
        this.x = e[t],
        this.y = e[t + 1],
        this.z = e[t + 2],
        this.w = e[t + 3],
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this.x,
        e[t + 1] = this.y,
        e[t + 2] = this.z,
        e[t + 3] = this.w,
        e
    },
    fromAttribute: function(e, t, i) {
        return void 0 === i && (i = 0),
        t = t * e.itemSize + i,
        this.x = e.array[t],
        this.y = e.array[t + 1],
        this.z = e.array[t + 2],
        this.w = e.array[t + 3],
        this
    }
},
n.Euler = function(e, t, i, r) {
    this._x = e || 0,
    this._y = t || 0,
    this._z = i || 0,
    this._order = r || n.Euler.DefaultOrder
}
,
n.Euler.RotationOrders = ["XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX"],
n.Euler.DefaultOrder = "XYZ",
n.Euler.prototype = {
    constructor: n.Euler,
    get x() {
        return this._x
    },
    set x(e) {
        this._x = e,
        this.onChangeCallback()
    },
    get y() {
        return this._y
    },
    set y(e) {
        this._y = e,
        this.onChangeCallback()
    },
    get z() {
        return this._z
    },
    set z(e) {
        this._z = e,
        this.onChangeCallback()
    },
    get order() {
        return this._order
    },
    set order(e) {
        this._order = e,
        this.onChangeCallback()
    },
    set: function(e, t, i, n) {
        return this._x = e,
        this._y = t,
        this._z = i,
        this._order = n || this._order,
        this.onChangeCallback(),
        this
    },
    clone: function() {
        return new this.constructor(this._x,this._y,this._z,this._order)
    },
    copy: function(e) {
        return this._x = e._x,
        this._y = e._y,
        this._z = e._z,
        this._order = e._order,
        this.onChangeCallback(),
        this
    },
    setFromRotationMatrix: function(e, t, i) {
        var r = n.Math.clamp
          , o = e.elements
          , a = o[0]
          , s = o[4]
          , l = o[8]
          , h = o[1]
          , c = o[5]
          , u = o[9]
          , d = o[2]
          , p = o[6]
          , f = o[10];
        return t = t || this._order,
        "XYZ" === t ? (this._y = Math.asin(r(l, -1, 1)),
        Math.abs(l) < .99999 ? (this._x = Math.atan2(-u, f),
        this._z = Math.atan2(-s, a)) : (this._x = Math.atan2(p, c),
        this._z = 0)) : "YXZ" === t ? (this._x = Math.asin(-r(u, -1, 1)),
        Math.abs(u) < .99999 ? (this._y = Math.atan2(l, f),
        this._z = Math.atan2(h, c)) : (this._y = Math.atan2(-d, a),
        this._z = 0)) : "ZXY" === t ? (this._x = Math.asin(r(p, -1, 1)),
        Math.abs(p) < .99999 ? (this._y = Math.atan2(-d, f),
        this._z = Math.atan2(-s, c)) : (this._y = 0,
        this._z = Math.atan2(h, a))) : "ZYX" === t ? (this._y = Math.asin(-r(d, -1, 1)),
        Math.abs(d) < .99999 ? (this._x = Math.atan2(p, f),
        this._z = Math.atan2(h, a)) : (this._x = 0,
        this._z = Math.atan2(-s, c))) : "YZX" === t ? (this._z = Math.asin(r(h, -1, 1)),
        Math.abs(h) < .99999 ? (this._x = Math.atan2(-u, c),
        this._y = Math.atan2(-d, a)) : (this._x = 0,
        this._y = Math.atan2(l, f))) : "XZY" === t ? (this._z = Math.asin(-r(s, -1, 1)),
        Math.abs(s) < .99999 ? (this._x = Math.atan2(p, c),
        this._y = Math.atan2(l, a)) : (this._x = Math.atan2(-u, f),
        this._y = 0)) : console.warn("THREE.Euler: .setFromRotationMatrix() given unsupported order: " + t),
        this._order = t,
        i !== !1 && this.onChangeCallback(),
        this
    },
    setFromQuaternion: function() {
        var e;
        return function(t, i, r) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationFromQuaternion(t),
            this.setFromRotationMatrix(e, i, r),
            this
        }
    }(),
    setFromVector3: function(e, t) {
        return this.set(e.x, e.y, e.z, t || this._order)
    },
    reorder: function() {
        var e = new n.Quaternion;
        return function(t) {
            e.setFromEuler(this),
            this.setFromQuaternion(e, t)
        }
    }(),
    equals: function(e) {
        return e._x === this._x && e._y === this._y && e._z === this._z && e._order === this._order
    },
    fromArray: function(e) {
        return this._x = e[0],
        this._y = e[1],
        this._z = e[2],
        void 0 !== e[3] && (this._order = e[3]),
        this.onChangeCallback(),
        this
    },
    toArray: function(e, t) {
        return void 0 === e && (e = []),
        void 0 === t && (t = 0),
        e[t] = this._x,
        e[t + 1] = this._y,
        e[t + 2] = this._z,
        e[t + 3] = this._order,
        e
    },
    toVector3: function(e) {
        return e ? e.set(this._x, this._y, this._z) : new n.Vector3(this._x,this._y,this._z)
    },
    onChange: function(e) {
        return this.onChangeCallback = e,
        this
    },
    onChangeCallback: function() {}
},
n.Line3 = function(e, t) {
    this.start = void 0 !== e ? e : new n.Vector3,
    this.end = void 0 !== t ? t : new n.Vector3
}
,
n.Line3.prototype = {
    constructor: n.Line3,
    set: function(e, t) {
        return this.start.copy(e),
        this.end.copy(t),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.start.copy(e.start),
        this.end.copy(e.end),
        this
    },
    center: function(e) {
        var t = e || new n.Vector3;
        return t.addVectors(this.start, this.end).multiplyScalar(.5)
    },
    delta: function(e) {
        var t = e || new n.Vector3;
        return t.subVectors(this.end, this.start)
    },
    distanceSq: function() {
        return this.start.distanceToSquared(this.end)
    },
    distance: function() {
        return this.start.distanceTo(this.end)
    },
    at: function(e, t) {
        var i = t || new n.Vector3;
        return this.delta(i).multiplyScalar(e).add(this.start)
    },
    closestPointToPointParameter: function() {
        var e = new n.Vector3
          , t = new n.Vector3;
        return function(i, r) {
            e.subVectors(i, this.start),
            t.subVectors(this.end, this.start);
            var o = t.dot(t)
              , a = t.dot(e)
              , s = a / o;
            return r && (s = n.Math.clamp(s, 0, 1)),
            s
        }
    }(),
    closestPointToPoint: function(e, t, i) {
        var r = this.closestPointToPointParameter(e, t)
          , o = i || new n.Vector3;
        return this.delta(o).multiplyScalar(r).add(this.start)
    },
    applyMatrix4: function(e) {
        return this.start.applyMatrix4(e),
        this.end.applyMatrix4(e),
        this
    },
    equals: function(e) {
        return e.start.equals(this.start) && e.end.equals(this.end)
    }
},
n.Box2 = function(e, t) {
    this.min = void 0 !== e ? e : new n.Vector2(+(1 / 0),+(1 / 0)),
    this.max = void 0 !== t ? t : new n.Vector2(-(1 / 0),-(1 / 0))
}
,
n.Box2.prototype = {
    constructor: n.Box2,
    set: function(e, t) {
        return this.min.copy(e),
        this.max.copy(t),
        this
    },
    setFromPoints: function(e) {
        this.makeEmpty();
        for (var t = 0, i = e.length; t < i; t++)
            this.expandByPoint(e[t]);
        return this
    },
    setFromCenterAndSize: function() {
        var e = new n.Vector2;
        return function(t, i) {
            var n = e.copy(i).multiplyScalar(.5);
            return this.min.copy(t).sub(n),
            this.max.copy(t).add(n),
            this
        }
    }(),
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.min.copy(e.min),
        this.max.copy(e.max),
        this
    },
    makeEmpty: function() {
        return this.min.x = this.min.y = +(1 / 0),
        this.max.x = this.max.y = -(1 / 0),
        this
    },
    isEmpty: function() {
        return this.max.x < this.min.x || this.max.y < this.min.y
    },
    center: function(e) {
        var t = e || new n.Vector2;
        return t.addVectors(this.min, this.max).multiplyScalar(.5)
    },
    size: function(e) {
        var t = e || new n.Vector2;
        return t.subVectors(this.max, this.min)
    },
    expandByPoint: function(e) {
        return this.min.min(e),
        this.max.max(e),
        this
    },
    expandByVector: function(e) {
        return this.min.sub(e),
        this.max.add(e),
        this
    },
    expandByScalar: function(e) {
        return this.min.addScalar(-e),
        this.max.addScalar(e),
        this
    },
    containsPoint: function(e) {
        return !(e.x < this.min.x || e.x > this.max.x || e.y < this.min.y || e.y > this.max.y)
    },
    containsBox: function(e) {
        return this.min.x <= e.min.x && e.max.x <= this.max.x && this.min.y <= e.min.y && e.max.y <= this.max.y
    },
    getParameter: function(e, t) {
        var i = t || new n.Vector2;
        return i.set((e.x - this.min.x) / (this.max.x - this.min.x), (e.y - this.min.y) / (this.max.y - this.min.y))
    },
    intersectsBox: function(e) {
        return !(e.max.x < this.min.x || e.min.x > this.max.x || e.max.y < this.min.y || e.min.y > this.max.y)
    },
    clampPoint: function(e, t) {
        var i = t || new n.Vector2;
        return i.copy(e).clamp(this.min, this.max)
    },
    distanceToPoint: function() {
        var e = new n.Vector2;
        return function(t) {
            var i = e.copy(t).clamp(this.min, this.max);
            return i.sub(t).length()
        }
    }(),
    intersect: function(e) {
        return this.min.max(e.min),
        this.max.min(e.max),
        this
    },
    union: function(e) {
        return this.min.min(e.min),
        this.max.max(e.max),
        this
    },
    translate: function(e) {
        return this.min.add(e),
        this.max.add(e),
        this
    },
    equals: function(e) {
        return e.min.equals(this.min) && e.max.equals(this.max)
    }
},
n.Box3 = function(e, t) {
    this.min = void 0 !== e ? e : new n.Vector3(+(1 / 0),+(1 / 0),+(1 / 0)),
    this.max = void 0 !== t ? t : new n.Vector3(-(1 / 0),-(1 / 0),-(1 / 0))
}
,
n.Box3.prototype = {
    constructor: n.Box3,
    set: function(e, t) {
        return this.min.copy(e),
        this.max.copy(t),
        this
    },
    setFromArray: function(e) {
        this.makeEmpty();
        for (var t = +(1 / 0), i = +(1 / 0), n = +(1 / 0), r = -(1 / 0), o = -(1 / 0), a = -(1 / 0), s = 0, l = e.length; s < l; s += 3) {
            var h = e[s]
              , c = e[s + 1]
              , u = e[s + 2];
            h < t && (t = h),
            c < i && (i = c),
            u < n && (n = u),
            h > r && (r = h),
            c > o && (o = c),
            u > a && (a = u)
        }
        this.min.set(t, i, n),
        this.max.set(r, o, a)
    },
    setFromPoints: function(e) {
        this.makeEmpty();
        for (var t = 0, i = e.length; t < i; t++)
            this.expandByPoint(e[t]);
        return this
    },
    setFromCenterAndSize: function() {
        var e = new n.Vector3;
        return function(t, i) {
            var n = e.copy(i).multiplyScalar(.5);
            return this.min.copy(t).sub(n),
            this.max.copy(t).add(n),
            this
        }
    }(),
    setFromObject: function() {
        var e;
        return function(t) {
            void 0 === e && (e = new n.Box3);
            var i = this;
            return this.makeEmpty(),
            t.updateMatrixWorld(!0),
            t.traverse(function(t) {
                var n = t.geometry;
                void 0 !== n && (null === n.boundingBox && n.computeBoundingBox(),
                n.boundingBox.isEmpty() === !1 && (e.copy(n.boundingBox),
                e.applyMatrix4(t.matrixWorld),
                i.union(e)))
            }),
            this
        }
    }(),
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.min.copy(e.min),
        this.max.copy(e.max),
        this
    },
    makeEmpty: function() {
        return this.min.x = this.min.y = this.min.z = +(1 / 0),
        this.max.x = this.max.y = this.max.z = -(1 / 0),
        this
    },
    isEmpty: function() {
        return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z
    },
    center: function(e) {
        var t = e || new n.Vector3;
        return t.addVectors(this.min, this.max).multiplyScalar(.5)
    },
    size: function(e) {
        var t = e || new n.Vector3;
        return t.subVectors(this.max, this.min)
    },
    expandByPoint: function(e) {
        return this.min.min(e),
        this.max.max(e),
        this
    },
    expandByVector: function(e) {
        return this.min.sub(e),
        this.max.add(e),
        this
    },
    expandByScalar: function(e) {
        return this.min.addScalar(-e),
        this.max.addScalar(e),
        this
    },
    containsPoint: function(e) {
        return !(e.x < this.min.x || e.x > this.max.x || e.y < this.min.y || e.y > this.max.y || e.z < this.min.z || e.z > this.max.z)
    },
    containsBox: function(e) {
        return this.min.x <= e.min.x && e.max.x <= this.max.x && this.min.y <= e.min.y && e.max.y <= this.max.y && this.min.z <= e.min.z && e.max.z <= this.max.z
    },
    getParameter: function(e, t) {
        var i = t || new n.Vector3;
        return i.set((e.x - this.min.x) / (this.max.x - this.min.x), (e.y - this.min.y) / (this.max.y - this.min.y), (e.z - this.min.z) / (this.max.z - this.min.z))
    },
    intersectsBox: function(e) {
        return !(e.max.x < this.min.x || e.min.x > this.max.x || e.max.y < this.min.y || e.min.y > this.max.y || e.max.z < this.min.z || e.min.z > this.max.z)
    },
    intersectsSphere: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Vector3),
            this.clampPoint(t.center, e),
            e.distanceToSquared(t.center) <= t.radius * t.radius
        }
    }(),
    intersectsPlane: function(e) {
        var t, i;
        return e.normal.x > 0 ? (t = e.normal.x * this.min.x,
        i = e.normal.x * this.max.x) : (t = e.normal.x * this.max.x,
        i = e.normal.x * this.min.x),
        e.normal.y > 0 ? (t += e.normal.y * this.min.y,
        i += e.normal.y * this.max.y) : (t += e.normal.y * this.max.y,
        i += e.normal.y * this.min.y),
        e.normal.z > 0 ? (t += e.normal.z * this.min.z,
        i += e.normal.z * this.max.z) : (t += e.normal.z * this.max.z,
        i += e.normal.z * this.min.z),
        t <= e.constant && i >= e.constant
    },
    clampPoint: function(e, t) {
        var i = t || new n.Vector3;
        return i.copy(e).clamp(this.min, this.max)
    },
    distanceToPoint: function() {
        var e = new n.Vector3;
        return function(t) {
            var i = e.copy(t).clamp(this.min, this.max);
            return i.sub(t).length()
        }
    }(),
    getBoundingSphere: function() {
        var e = new n.Vector3;
        return function(t) {
            var i = t || new n.Sphere;
            return i.center = this.center(),
            i.radius = .5 * this.size(e).length(),
            i
        }
    }(),
    intersect: function(e) {
        return this.min.max(e.min),
        this.max.min(e.max),
        this
    },
    union: function(e) {
        return this.min.min(e.min),
        this.max.max(e.max),
        this
    },
    applyMatrix4: function() {
        var e = [new n.Vector3, new n.Vector3, new n.Vector3, new n.Vector3, new n.Vector3, new n.Vector3, new n.Vector3, new n.Vector3];
        return function(t) {
            return e[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(t),
            e[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(t),
            e[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(t),
            e[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(t),
            e[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(t),
            e[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(t),
            e[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(t),
            e[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(t),
            this.makeEmpty(),
            this.setFromPoints(e),
            this
        }
    }(),
    translate: function(e) {
        return this.min.add(e),
        this.max.add(e),
        this
    },
    equals: function(e) {
        return e.min.equals(this.min) && e.max.equals(this.max)
    }
},
n.Matrix3 = function() {
    this.elements = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    arguments.length > 0 && console.error("THREE.Matrix3: the constructor no longer reads arguments. use .set() instead.")
}
,
n.Matrix3.prototype = {
    constructor: n.Matrix3,
    set: function(e, t, i, n, r, o, a, s, l) {
        var h = this.elements;
        return h[0] = e,
        h[1] = n,
        h[2] = a,
        h[3] = t,
        h[4] = r,
        h[5] = s,
        h[6] = i,
        h[7] = o,
        h[8] = l,
        this
    },
    identity: function() {
        return this.set(1, 0, 0, 0, 1, 0, 0, 0, 1),
        this
    },
    clone: function() {
        return (new this.constructor).fromArray(this.elements)
    },
    copy: function(e) {
        var t = e.elements;
        return this.set(t[0], t[3], t[6], t[1], t[4], t[7], t[2], t[5], t[8]),
        this
    },
    setFromMatrix4: function(e) {
        var t = e.elements;
        return this.set(t[0], t[4], t[8], t[1], t[5], t[9], t[2], t[6], t[10]),
        this
    },
    applyToVector3Array: function() {
        var e;
        return function(t, i, r) {
            void 0 === e && (e = new n.Vector3),
            void 0 === i && (i = 0),
            void 0 === r && (r = t.length);
            for (var o = 0, a = i; o < r; o += 3,
            a += 3)
                e.fromArray(t, a),
                e.applyMatrix3(this),
                e.toArray(t, a);
            return t
        }
    }(),
    applyToBuffer: function() {
        var e;
        return function(t, i, r) {
            void 0 === e && (e = new n.Vector3),
            void 0 === i && (i = 0),
            void 0 === r && (r = t.length / t.itemSize);
            for (var o = 0, a = i; o < r; o++,
            a++)
                e.x = t.getX(a),
                e.y = t.getY(a),
                e.z = t.getZ(a),
                e.applyMatrix3(this),
                t.setXYZ(e.x, e.y, e.z);
            return t
        }
    }(),
    multiplyScalar: function(e) {
        var t = this.elements;
        return t[0] *= e,
        t[3] *= e,
        t[6] *= e,
        t[1] *= e,
        t[4] *= e,
        t[7] *= e,
        t[2] *= e,
        t[5] *= e,
        t[8] *= e,
        this
    },
    determinant: function() {
        var e = this.elements
          , t = e[0]
          , i = e[1]
          , n = e[2]
          , r = e[3]
          , o = e[4]
          , a = e[5]
          , s = e[6]
          , l = e[7]
          , h = e[8];
        return t * o * h - t * a * l - i * r * h + i * a * s + n * r * l - n * o * s
    },
    getInverse: function(e, t) {
        e instanceof n.Matrix4 && console.warn("THREE.Matrix3.getInverse no longer takes a Matrix4 argument.");
        var i = e.elements
          , r = this.elements
          , o = i[0]
          , a = i[1]
          , s = i[2]
          , l = i[3]
          , h = i[4]
          , c = i[5]
          , u = i[6]
          , d = i[7]
          , p = i[8]
          , f = p * h - c * d
          , g = c * u - p * l
          , m = d * l - h * u
          , v = o * f + a * g + s * m;
        if (0 === v) {
            var A = "THREE.Matrix3.getInverse(): can't invert matrix, determinant is 0";
            if (t)
                throw new Error(A);
            return console.warn(A),
            this.identity()
        }
        return r[0] = f,
        r[1] = s * d - p * a,
        r[2] = c * a - s * h,
        r[3] = g,
        r[4] = p * o - s * u,
        r[5] = s * l - c * o,
        r[6] = m,
        r[7] = a * u - d * o,
        r[8] = h * o - a * l,
        this.multiplyScalar(1 / v)
    },
    transpose: function() {
        var e, t = this.elements;
        return e = t[1],
        t[1] = t[3],
        t[3] = e,
        e = t[2],
        t[2] = t[6],
        t[6] = e,
        e = t[5],
        t[5] = t[7],
        t[7] = e,
        this
    },
    flattenToArrayOffset: function(e, t) {
        var i = this.elements;
        return e[t] = i[0],
        e[t + 1] = i[1],
        e[t + 2] = i[2],
        e[t + 3] = i[3],
        e[t + 4] = i[4],
        e[t + 5] = i[5],
        e[t + 6] = i[6],
        e[t + 7] = i[7],
        e[t + 8] = i[8],
        e
    },
    getNormalMatrix: function(e) {
        return this.setFromMatrix4(e).getInverse(this).transpose()
    },
    transposeIntoArray: function(e) {
        var t = this.elements;
        return e[0] = t[0],
        e[1] = t[3],
        e[2] = t[6],
        e[3] = t[1],
        e[4] = t[4],
        e[5] = t[7],
        e[6] = t[2],
        e[7] = t[5],
        e[8] = t[8],
        this
    },
    fromArray: function(e) {
        return this.elements.set(e),
        this
    },
    toArray: function() {
        var e = this.elements;
        return [e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]]
    }
},
n.Matrix4 = function() {
    this.elements = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    arguments.length > 0 && console.error("THREE.Matrix4: the constructor no longer reads arguments. use .set() instead.")
}
,
n.Matrix4.prototype = {
    constructor: n.Matrix4,
    set: function(e, t, i, n, r, o, a, s, l, h, c, u, d, p, f, g) {
        var m = this.elements;
        return m[0] = e,
        m[4] = t,
        m[8] = i,
        m[12] = n,
        m[1] = r,
        m[5] = o,
        m[9] = a,
        m[13] = s,
        m[2] = l,
        m[6] = h,
        m[10] = c,
        m[14] = u,
        m[3] = d,
        m[7] = p,
        m[11] = f,
        m[15] = g,
        this
    },
    identity: function() {
        return this.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        this
    },
    clone: function() {
        return (new n.Matrix4).fromArray(this.elements)
    },
    copy: function(e) {
        return this.elements.set(e.elements),
        this
    },
    copyPosition: function(e) {
        var t = this.elements
          , i = e.elements;
        return t[12] = i[12],
        t[13] = i[13],
        t[14] = i[14],
        this
    },
    extractBasis: function(e, t, i) {
        return e.setFromMatrixColumn(this, 0),
        t.setFromMatrixColumn(this, 1),
        i.setFromMatrixColumn(this, 2),
        this
    },
    makeBasis: function(e, t, i) {
        return this.set(e.x, t.x, i.x, 0, e.y, t.y, i.y, 0, e.z, t.z, i.z, 0, 0, 0, 0, 1),
        this
    },
    extractRotation: function() {
        var e;
        return function(t) {
            void 0 === e && (e = new n.Vector3);
            var i = this.elements
              , r = t.elements
              , o = 1 / e.setFromMatrixColumn(t, 0).length()
              , a = 1 / e.setFromMatrixColumn(t, 1).length()
              , s = 1 / e.setFromMatrixColumn(t, 2).length();
            return i[0] = r[0] * o,
            i[1] = r[1] * o,
            i[2] = r[2] * o,
            i[4] = r[4] * a,
            i[5] = r[5] * a,
            i[6] = r[6] * a,
            i[8] = r[8] * s,
            i[9] = r[9] * s,
            i[10] = r[10] * s,
            this
        }
    }(),
    makeRotationFromEuler: function(e) {
        e instanceof n.Euler == !1 && console.error("THREE.Matrix: .makeRotationFromEuler() now expects a Euler rotation rather than a Vector3 and order.");
        var t = this.elements
          , i = e.x
          , r = e.y
          , o = e.z
          , a = Math.cos(i)
          , s = Math.sin(i)
          , l = Math.cos(r)
          , h = Math.sin(r)
          , c = Math.cos(o)
          , u = Math.sin(o);
        if ("XYZ" === e.order) {
            var d = a * c
              , p = a * u
              , f = s * c
              , g = s * u;
            t[0] = l * c,
            t[4] = -l * u,
            t[8] = h,
            t[1] = p + f * h,
            t[5] = d - g * h,
            t[9] = -s * l,
            t[2] = g - d * h,
            t[6] = f + p * h,
            t[10] = a * l
        } else if ("YXZ" === e.order) {
            var m = l * c
              , v = l * u
              , A = h * c
              , y = h * u;
            t[0] = m + y * s,
            t[4] = A * s - v,
            t[8] = a * h,
            t[1] = a * u,
            t[5] = a * c,
            t[9] = -s,
            t[2] = v * s - A,
            t[6] = y + m * s,
            t[10] = a * l
        } else if ("ZXY" === e.order) {
            var m = l * c
              , v = l * u
              , A = h * c
              , y = h * u;
            t[0] = m - y * s,
            t[4] = -a * u,
            t[8] = A + v * s,
            t[1] = v + A * s,
            t[5] = a * c,
            t[9] = y - m * s,
            t[2] = -a * h,
            t[6] = s,
            t[10] = a * l
        } else if ("ZYX" === e.order) {
            var d = a * c
              , p = a * u
              , f = s * c
              , g = s * u;
            t[0] = l * c,
            t[4] = f * h - p,
            t[8] = d * h + g,
            t[1] = l * u,
            t[5] = g * h + d,
            t[9] = p * h - f,
            t[2] = -h,
            t[6] = s * l,
            t[10] = a * l
        } else if ("YZX" === e.order) {
            var C = a * l
              , I = a * h
              , b = s * l
              , w = s * h;
            t[0] = l * c,
            t[4] = w - C * u,
            t[8] = b * u + I,
            t[1] = u,
            t[5] = a * c,
            t[9] = -s * c,
            t[2] = -h * c,
            t[6] = I * u + b,
            t[10] = C - w * u
        } else if ("XZY" === e.order) {
            var C = a * l
              , I = a * h
              , b = s * l
              , w = s * h;
            t[0] = l * c,
            t[4] = -u,
            t[8] = h * c,
            t[1] = C * u + w,
            t[5] = a * c,
            t[9] = I * u - b,
            t[2] = b * u - I,
            t[6] = s * c,
            t[10] = w * u + C
        }
        return t[3] = 0,
        t[7] = 0,
        t[11] = 0,
        t[12] = 0,
        t[13] = 0,
        t[14] = 0,
        t[15] = 1,
        this
    },
    makeRotationFromQuaternion: function(e) {
        var t = this.elements
          , i = e.x
          , n = e.y
          , r = e.z
          , o = e.w
          , a = i + i
          , s = n + n
          , l = r + r
          , h = i * a
          , c = i * s
          , u = i * l
          , d = n * s
          , p = n * l
          , f = r * l
          , g = o * a
          , m = o * s
          , v = o * l;
        return t[0] = 1 - (d + f),
        t[4] = c - v,
        t[8] = u + m,
        t[1] = c + v,
        t[5] = 1 - (h + f),
        t[9] = p - g,
        t[2] = u - m,
        t[6] = p + g,
        t[10] = 1 - (h + d),
        t[3] = 0,
        t[7] = 0,
        t[11] = 0,
        t[12] = 0,
        t[13] = 0,
        t[14] = 0,
        t[15] = 1,
        this
    },
    lookAt: function() {
        var e, t, i;
        return function(r, o, a) {
            void 0 === e && (e = new n.Vector3),
            void 0 === t && (t = new n.Vector3),
            void 0 === i && (i = new n.Vector3);
            var s = this.elements;
            return i.subVectors(r, o).normalize(),
            0 === i.lengthSq() && (i.z = 1),
            e.crossVectors(a, i).normalize(),
            0 === e.lengthSq() && (i.x += 1e-4,
            e.crossVectors(a, i).normalize()),
            t.crossVectors(i, e),
            s[0] = e.x,
            s[4] = t.x,
            s[8] = i.x,
            s[1] = e.y,
            s[5] = t.y,
            s[9] = i.y,
            s[2] = e.z,
            s[6] = t.z,
            s[10] = i.z,
            this
        }
    }(),
    multiply: function(e, t) {
        return void 0 !== t ? (console.warn("THREE.Matrix4: .multiply() now only accepts one argument. Use .multiplyMatrices( a, b ) instead."),
        this.multiplyMatrices(e, t)) : this.multiplyMatrices(this, e)
    },
    multiplyMatrices: function(e, t) {
        var i = e.elements
          , n = t.elements
          , r = this.elements
          , o = i[0]
          , a = i[4]
          , s = i[8]
          , l = i[12]
          , h = i[1]
          , c = i[5]
          , u = i[9]
          , d = i[13]
          , p = i[2]
          , f = i[6]
          , g = i[10]
          , m = i[14]
          , v = i[3]
          , A = i[7]
          , y = i[11]
          , C = i[15]
          , I = n[0]
          , b = n[4]
          , w = n[8]
          , E = n[12]
          , x = n[1]
          , T = n[5]
          , M = n[9]
          , S = n[13]
          , _ = n[2]
          , P = n[6]
          , R = n[10]
          , L = n[14]
          , O = n[3]
          , D = n[7]
          , F = n[11]
          , N = n[15];
        return r[0] = o * I + a * x + s * _ + l * O,
        r[4] = o * b + a * T + s * P + l * D,
        r[8] = o * w + a * M + s * R + l * F,
        r[12] = o * E + a * S + s * L + l * N,
        r[1] = h * I + c * x + u * _ + d * O,
        r[5] = h * b + c * T + u * P + d * D,
        r[9] = h * w + c * M + u * R + d * F,
        r[13] = h * E + c * S + u * L + d * N,
        r[2] = p * I + f * x + g * _ + m * O,
        r[6] = p * b + f * T + g * P + m * D,
        r[10] = p * w + f * M + g * R + m * F,
        r[14] = p * E + f * S + g * L + m * N,
        r[3] = v * I + A * x + y * _ + C * O,
        r[7] = v * b + A * T + y * P + C * D,
        r[11] = v * w + A * M + y * R + C * F,
        r[15] = v * E + A * S + y * L + C * N,
        this
    },
    multiplyToArray: function(e, t, i) {
        var n = this.elements;
        return this.multiplyMatrices(e, t),
        i[0] = n[0],
        i[1] = n[1],
        i[2] = n[2],
        i[3] = n[3],
        i[4] = n[4],
        i[5] = n[5],
        i[6] = n[6],
        i[7] = n[7],
        i[8] = n[8],
        i[9] = n[9],
        i[10] = n[10],
        i[11] = n[11],
        i[12] = n[12],
        i[13] = n[13],
        i[14] = n[14],
        i[15] = n[15],
        this
    },
    multiplyScalar: function(e) {
        var t = this.elements;
        return t[0] *= e,
        t[4] *= e,
        t[8] *= e,
        t[12] *= e,
        t[1] *= e,
        t[5] *= e,
        t[9] *= e,
        t[13] *= e,
        t[2] *= e,
        t[6] *= e,
        t[10] *= e,
        t[14] *= e,
        t[3] *= e,
        t[7] *= e,
        t[11] *= e,
        t[15] *= e,
        this
    },
    applyToVector3Array: function() {
        var e;
        return function(t, i, r) {
            void 0 === e && (e = new n.Vector3),
            void 0 === i && (i = 0),
            void 0 === r && (r = t.length);
            for (var o = 0, a = i; o < r; o += 3,
            a += 3)
                e.fromArray(t, a),
                e.applyMatrix4(this),
                e.toArray(t, a);
            return t
        }
    }(),
    applyToBuffer: function() {
        var e;
        return function(t, i, r) {
            void 0 === e && (e = new n.Vector3),
            void 0 === i && (i = 0),
            void 0 === r && (r = t.length / t.itemSize);
            for (var o = 0, a = i; o < r; o++,
            a++)
                e.x = t.getX(a),
                e.y = t.getY(a),
                e.z = t.getZ(a),
                e.applyMatrix4(this),
                t.setXYZ(e.x, e.y, e.z);
            return t
        }
    }(),
    determinant: function() {
        var e = this.elements
          , t = e[0]
          , i = e[4]
          , n = e[8]
          , r = e[12]
          , o = e[1]
          , a = e[5]
          , s = e[9]
          , l = e[13]
          , h = e[2]
          , c = e[6]
          , u = e[10]
          , d = e[14]
          , p = e[3]
          , f = e[7]
          , g = e[11]
          , m = e[15];
        return p * (+r * s * c - n * l * c - r * a * u + i * l * u + n * a * d - i * s * d) + f * (+t * s * d - t * l * u + r * o * u - n * o * d + n * l * h - r * s * h) + g * (+t * l * c - t * a * d - r * o * c + i * o * d + r * a * h - i * l * h) + m * (-n * a * h - t * s * c + t * a * u + n * o * c - i * o * u + i * s * h)
    },
    transpose: function() {
        var e, t = this.elements;
        return e = t[1],
        t[1] = t[4],
        t[4] = e,
        e = t[2],
        t[2] = t[8],
        t[8] = e,
        e = t[6],
        t[6] = t[9],
        t[9] = e,
        e = t[3],
        t[3] = t[12],
        t[12] = e,
        e = t[7],
        t[7] = t[13],
        t[13] = e,
        e = t[11],
        t[11] = t[14],
        t[14] = e,
        this
    },
    flattenToArrayOffset: function(e, t) {
        var i = this.elements;
        return e[t] = i[0],
        e[t + 1] = i[1],
        e[t + 2] = i[2],
        e[t + 3] = i[3],
        e[t + 4] = i[4],
        e[t + 5] = i[5],
        e[t + 6] = i[6],
        e[t + 7] = i[7],
        e[t + 8] = i[8],
        e[t + 9] = i[9],
        e[t + 10] = i[10],
        e[t + 11] = i[11],
        e[t + 12] = i[12],
        e[t + 13] = i[13],
        e[t + 14] = i[14],
        e[t + 15] = i[15],
        e
    },
    getPosition: function() {
        var e;
        return function() {
            return void 0 === e && (e = new n.Vector3),
            console.warn("THREE.Matrix4: .getPosition() has been removed. Use Vector3.setFromMatrixPosition( matrix ) instead."),
            e.setFromMatrixColumn(this, 3)
        }
    }(),
    setPosition: function(e) {
        var t = this.elements;
        return t[12] = e.x,
        t[13] = e.y,
        t[14] = e.z,
        this
    },
    getInverse: function(e, t) {
        var i = this.elements
          , n = e.elements
          , r = n[0]
          , o = n[1]
          , a = n[2]
          , s = n[3]
          , l = n[4]
          , h = n[5]
          , c = n[6]
          , u = n[7]
          , d = n[8]
          , p = n[9]
          , f = n[10]
          , g = n[11]
          , m = n[12]
          , v = n[13]
          , A = n[14]
          , y = n[15]
          , C = p * A * u - v * f * u + v * c * g - h * A * g - p * c * y + h * f * y
          , I = m * f * u - d * A * u - m * c * g + l * A * g + d * c * y - l * f * y
          , b = d * v * u - m * p * u + m * h * g - l * v * g - d * h * y + l * p * y
          , w = m * p * c - d * v * c - m * h * f + l * v * f + d * h * A - l * p * A
          , E = r * C + o * I + a * b + s * w;
        if (0 === E) {
            var x = "THREE.Matrix4.getInverse(): can't invert matrix, determinant is 0";
            if (t)
                throw new Error(x);
            return console.warn(x),
            this.identity()
        }
        return i[0] = C,
        i[1] = v * f * s - p * A * s - v * a * g + o * A * g + p * a * y - o * f * y,
        i[2] = h * A * s - v * c * s + v * a * u - o * A * u - h * a * y + o * c * y,
        i[3] = p * c * s - h * f * s - p * a * u + o * f * u + h * a * g - o * c * g,
        i[4] = I,
        i[5] = d * A * s - m * f * s + m * a * g - r * A * g - d * a * y + r * f * y,
        i[6] = m * c * s - l * A * s - m * a * u + r * A * u + l * a * y - r * c * y,
        i[7] = l * f * s - d * c * s + d * a * u - r * f * u - l * a * g + r * c * g,
        i[8] = b,
        i[9] = m * p * s - d * v * s - m * o * g + r * v * g + d * o * y - r * p * y,
        i[10] = l * v * s - m * h * s + m * o * u - r * v * u - l * o * y + r * h * y,
        i[11] = d * h * s - l * p * s - d * o * u + r * p * u + l * o * g - r * h * g,
        i[12] = w,
        i[13] = d * v * a - m * p * a + m * o * f - r * v * f - d * o * A + r * p * A,
        i[14] = m * h * a - l * v * a - m * o * c + r * v * c + l * o * A - r * h * A,
        i[15] = l * p * a - d * h * a + d * o * c - r * p * c - l * o * f + r * h * f,
        this.multiplyScalar(1 / E)
    },
    scale: function(e) {
        var t = this.elements
          , i = e.x
          , n = e.y
          , r = e.z;
        return t[0] *= i,
        t[4] *= n,
        t[8] *= r,
        t[1] *= i,
        t[5] *= n,
        t[9] *= r,
        t[2] *= i,
        t[6] *= n,
        t[10] *= r,
        t[3] *= i,
        t[7] *= n,
        t[11] *= r,
        this
    },
    getMaxScaleOnAxis: function() {
        var e = this.elements
          , t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
          , i = e[4] * e[4] + e[5] * e[5] + e[6] * e[6]
          , n = e[8] * e[8] + e[9] * e[9] + e[10] * e[10];
        return Math.sqrt(Math.max(t, i, n))
    },
    makeTranslation: function(e, t, i) {
        return this.set(1, 0, 0, e, 0, 1, 0, t, 0, 0, 1, i, 0, 0, 0, 1),
        this
    },
    makeRotationX: function(e) {
        var t = Math.cos(e)
          , i = Math.sin(e);
        return this.set(1, 0, 0, 0, 0, t, -i, 0, 0, i, t, 0, 0, 0, 0, 1),
        this
    },
    makeRotationY: function(e) {
        var t = Math.cos(e)
          , i = Math.sin(e);
        return this.set(t, 0, i, 0, 0, 1, 0, 0, -i, 0, t, 0, 0, 0, 0, 1),
        this
    },
    makeRotationZ: function(e) {
        var t = Math.cos(e)
          , i = Math.sin(e);
        return this.set(t, -i, 0, 0, i, t, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        this
    },
    makeRotationAxis: function(e, t) {
        var i = Math.cos(t)
          , n = Math.sin(t)
          , r = 1 - i
          , o = e.x
          , a = e.y
          , s = e.z
          , l = r * o
          , h = r * a;
        return this.set(l * o + i, l * a - n * s, l * s + n * a, 0, l * a + n * s, h * a + i, h * s - n * o, 0, l * s - n * a, h * s + n * o, r * s * s + i, 0, 0, 0, 0, 1),
        this
    },
    makeScale: function(e, t, i) {
        return this.set(e, 0, 0, 0, 0, t, 0, 0, 0, 0, i, 0, 0, 0, 0, 1),
        this
    },
    compose: function(e, t, i) {
        return this.makeRotationFromQuaternion(t),
        this.scale(i),
        this.setPosition(e),
        this
    },
    decompose: function() {
        var e, t;
        return function(i, r, o) {
            void 0 === e && (e = new n.Vector3),
            void 0 === t && (t = new n.Matrix4);
            var a = this.elements
              , s = e.set(a[0], a[1], a[2]).length()
              , l = e.set(a[4], a[5], a[6]).length()
              , h = e.set(a[8], a[9], a[10]).length()
              , c = this.determinant();
            c < 0 && (s = -s),
            i.x = a[12],
            i.y = a[13],
            i.z = a[14],
            t.elements.set(this.elements);
            var u = 1 / s
              , d = 1 / l
              , p = 1 / h;
            return t.elements[0] *= u,
            t.elements[1] *= u,
            t.elements[2] *= u,
            t.elements[4] *= d,
            t.elements[5] *= d,
            t.elements[6] *= d,
            t.elements[8] *= p,
            t.elements[9] *= p,
            t.elements[10] *= p,
            r.setFromRotationMatrix(t),
            o.x = s,
            o.y = l,
            o.z = h,
            this
        }
    }(),
    makeFrustum: function(e, t, i, n, r, o) {
        var a = this.elements
          , s = 2 * r / (t - e)
          , l = 2 * r / (n - i)
          , h = (t + e) / (t - e)
          , c = (n + i) / (n - i)
          , u = -(o + r) / (o - r)
          , d = -2 * o * r / (o - r);
        return a[0] = s,
        a[4] = 0,
        a[8] = h,
        a[12] = 0,
        a[1] = 0,
        a[5] = l,
        a[9] = c,
        a[13] = 0,
        a[2] = 0,
        a[6] = 0,
        a[10] = u,
        a[14] = d,
        a[3] = 0,
        a[7] = 0,
        a[11] = -1,
        a[15] = 0,
        this
    },
    makePerspective: function(e, t, i, r) {
        var o = i * Math.tan(n.Math.degToRad(.5 * e))
          , a = -o
          , s = a * t
          , l = o * t;
        return this.makeFrustum(s, l, a, o, i, r)
    },
    makeOrthographic: function(e, t, i, n, r, o) {
        var a = this.elements
          , s = 1 / (t - e)
          , l = 1 / (i - n)
          , h = 1 / (o - r)
          , c = (t + e) * s
          , u = (i + n) * l
          , d = (o + r) * h;
        return a[0] = 2 * s,
        a[4] = 0,
        a[8] = 0,
        a[12] = -c,
        a[1] = 0,
        a[5] = 2 * l,
        a[9] = 0,
        a[13] = -u,
        a[2] = 0,
        a[6] = 0,
        a[10] = -2 * h,
        a[14] = -d,
        a[3] = 0,
        a[7] = 0,
        a[11] = 0,
        a[15] = 1,
        this
    },
    equals: function(e) {
        for (var t = this.elements, i = e.elements, n = 0; n < 16; n++)
            if (t[n] !== i[n])
                return !1;
        return !0
    },
    fromArray: function(e) {
        return this.elements.set(e),
        this
    },
    toArray: function() {
        var e = this.elements;
        return [e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15]]
    }
},
n.Ray = function(e, t) {
    this.origin = void 0 !== e ? e : new n.Vector3,
    this.direction = void 0 !== t ? t : new n.Vector3
}
,
n.Ray.prototype = {
    constructor: n.Ray,
    set: function(e, t) {
        return this.origin.copy(e),
        this.direction.copy(t),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.origin.copy(e.origin),
        this.direction.copy(e.direction),
        this
    },
    at: function(e, t) {
        var i = t || new n.Vector3;
        return i.copy(this.direction).multiplyScalar(e).add(this.origin)
    },
    lookAt: function(e) {
        this.direction.copy(e).sub(this.origin).normalize()
    },
    recast: function() {
        var e = new n.Vector3;
        return function(t) {
            return this.origin.copy(this.at(t, e)),
            this
        }
    }(),
    closestPointToPoint: function(e, t) {
        var i = t || new n.Vector3;
        i.subVectors(e, this.origin);
        var r = i.dot(this.direction);
        return r < 0 ? i.copy(this.origin) : i.copy(this.direction).multiplyScalar(r).add(this.origin)
    },
    distanceToPoint: function(e) {
        return Math.sqrt(this.distanceSqToPoint(e))
    },
    distanceSqToPoint: function() {
        var e = new n.Vector3;
        return function(t) {
            var i = e.subVectors(t, this.origin).dot(this.direction);
            return i < 0 ? this.origin.distanceToSquared(t) : (e.copy(this.direction).multiplyScalar(i).add(this.origin),
            e.distanceToSquared(t))
        }
    }(),
    distanceSqToSegment: function() {
        var e = new n.Vector3
          , t = new n.Vector3
          , i = new n.Vector3;
        return function(n, r, o, a) {
            e.copy(n).add(r).multiplyScalar(.5),
            t.copy(r).sub(n).normalize(),
            i.copy(this.origin).sub(e);
            var s, l, h, c, u = .5 * n.distanceTo(r), d = -this.direction.dot(t), p = i.dot(this.direction), f = -i.dot(t), g = i.lengthSq(), m = Math.abs(1 - d * d);
            if (m > 0)
                if (s = d * f - p,
                l = d * p - f,
                c = u * m,
                s >= 0)
                    if (l >= -c)
                        if (l <= c) {
                            var v = 1 / m;
                            s *= v,
                            l *= v,
                            h = s * (s + d * l + 2 * p) + l * (d * s + l + 2 * f) + g
                        } else
                            l = u,
                            s = Math.max(0, -(d * l + p)),
                            h = -s * s + l * (l + 2 * f) + g;
                    else
                        l = -u,
                        s = Math.max(0, -(d * l + p)),
                        h = -s * s + l * (l + 2 * f) + g;
                else
                    l <= -c ? (s = Math.max(0, -(-d * u + p)),
                    l = s > 0 ? -u : Math.min(Math.max(-u, -f), u),
                    h = -s * s + l * (l + 2 * f) + g) : l <= c ? (s = 0,
                    l = Math.min(Math.max(-u, -f), u),
                    h = l * (l + 2 * f) + g) : (s = Math.max(0, -(d * u + p)),
                    l = s > 0 ? u : Math.min(Math.max(-u, -f), u),
                    h = -s * s + l * (l + 2 * f) + g);
            else
                l = d > 0 ? -u : u,
                s = Math.max(0, -(d * l + p)),
                h = -s * s + l * (l + 2 * f) + g;
            return o && o.copy(this.direction).multiplyScalar(s).add(this.origin),
            a && a.copy(t).multiplyScalar(l).add(e),
            h
        }
    }(),
    intersectSphere: function() {
        var e = new n.Vector3;
        return function(t, i) {
            e.subVectors(t.center, this.origin);
            var n = e.dot(this.direction)
              , r = e.dot(e) - n * n
              , o = t.radius * t.radius;
            if (r > o)
                return null;
            var a = Math.sqrt(o - r)
              , s = n - a
              , l = n + a;
            return s < 0 && l < 0 ? null : s < 0 ? this.at(l, i) : this.at(s, i)
        }
    }(),
    intersectsSphere: function(e) {
        return this.distanceToPoint(e.center) <= e.radius
    },
    distanceToPlane: function(e) {
        var t = e.normal.dot(this.direction);
        if (0 === t)
            return 0 === e.distanceToPoint(this.origin) ? 0 : null;
        var i = -(this.origin.dot(e.normal) + e.constant) / t;
        return i >= 0 ? i : null
    },
    intersectPlane: function(e, t) {
        var i = this.distanceToPlane(e);
        return null === i ? null : this.at(i, t)
    },
    intersectsPlane: function(e) {
        var t = e.distanceToPoint(this.origin);
        if (0 === t)
            return !0;
        var i = e.normal.dot(this.direction);
        return i * t < 0
    },
    intersectBox: function(e, t) {
        var i, n, r, o, a, s, l = 1 / this.direction.x, h = 1 / this.direction.y, c = 1 / this.direction.z, u = this.origin;
        return l >= 0 ? (i = (e.min.x - u.x) * l,
        n = (e.max.x - u.x) * l) : (i = (e.max.x - u.x) * l,
        n = (e.min.x - u.x) * l),
        h >= 0 ? (r = (e.min.y - u.y) * h,
        o = (e.max.y - u.y) * h) : (r = (e.max.y - u.y) * h,
        o = (e.min.y - u.y) * h),
        i > o || r > n ? null : ((r > i || i !== i) && (i = r),
        (o < n || n !== n) && (n = o),
        c >= 0 ? (a = (e.min.z - u.z) * c,
        s = (e.max.z - u.z) * c) : (a = (e.max.z - u.z) * c,
        s = (e.min.z - u.z) * c),
        i > s || a > n ? null : ((a > i || i !== i) && (i = a),
        (s < n || n !== n) && (n = s),
        n < 0 ? null : this.at(i >= 0 ? i : n, t)))
    },
    intersectsBox: function() {
        var e = new n.Vector3;
        return function(t) {
            return null !== this.intersectBox(t, e)
        }
    }(),
    intersectTriangle: function() {
        var e = new n.Vector3
          , t = new n.Vector3
          , i = new n.Vector3
          , r = new n.Vector3;
        return function(n, o, a, s, l) {
            t.subVectors(o, n),
            i.subVectors(a, n),
            r.crossVectors(t, i);
            var h, c = this.direction.dot(r);
            if (c > 0) {
                if (s)
                    return null;
                h = 1
            } else {
                if (!(c < 0))
                    return null;
                h = -1,
                c = -c
            }
            e.subVectors(this.origin, n);
            var u = h * this.direction.dot(i.crossVectors(e, i));
            if (u < 0)
                return null;
            var d = h * this.direction.dot(t.cross(e));
            if (d < 0)
                return null;
            if (u + d > c)
                return null;
            var p = -h * e.dot(r);
            return p < 0 ? null : this.at(p / c, l)
        }
    }(),
    applyMatrix4: function(e) {
        return this.direction.add(this.origin).applyMatrix4(e),
        this.origin.applyMatrix4(e),
        this.direction.sub(this.origin),
        this.direction.normalize(),
        this
    },
    equals: function(e) {
        return e.origin.equals(this.origin) && e.direction.equals(this.direction)
    }
},
n.Sphere = function(e, t) {
    this.center = void 0 !== e ? e : new n.Vector3,
    this.radius = void 0 !== t ? t : 0
}
,
n.Sphere.prototype = {
    constructor: n.Sphere,
    set: function(e, t) {
        return this.center.copy(e),
        this.radius = t,
        this
    },
    setFromPoints: function() {
        var e = new n.Box3;
        return function(t, i) {
            var n = this.center;
            void 0 !== i ? n.copy(i) : e.setFromPoints(t).center(n);
            for (var r = 0, o = 0, a = t.length; o < a; o++)
                r = Math.max(r, n.distanceToSquared(t[o]));
            return this.radius = Math.sqrt(r),
            this
        }
    }(),
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.center.copy(e.center),
        this.radius = e.radius,
        this
    },
    empty: function() {
        return this.radius <= 0
    },
    containsPoint: function(e) {
        return e.distanceToSquared(this.center) <= this.radius * this.radius
    },
    distanceToPoint: function(e) {
        return e.distanceTo(this.center) - this.radius
    },
    intersectsSphere: function(e) {
        var t = this.radius + e.radius;
        return e.center.distanceToSquared(this.center) <= t * t
    },
    intersectsBox: function(e) {
        return e.intersectsSphere(this)
    },
    intersectsPlane: function(e) {
        return Math.abs(this.center.dot(e.normal) - e.constant) <= this.radius
    },
    clampPoint: function(e, t) {
        var i = this.center.distanceToSquared(e)
          , r = t || new n.Vector3;
        return r.copy(e),
        i > this.radius * this.radius && (r.sub(this.center).normalize(),
        r.multiplyScalar(this.radius).add(this.center)),
        r
    },
    getBoundingBox: function(e) {
        var t = e || new n.Box3;
        return t.set(this.center, this.center),
        t.expandByScalar(this.radius),
        t
    },
    applyMatrix4: function(e) {
        return this.center.applyMatrix4(e),
        this.radius = this.radius * e.getMaxScaleOnAxis(),
        this
    },
    translate: function(e) {
        return this.center.add(e),
        this
    },
    equals: function(e) {
        return e.center.equals(this.center) && e.radius === this.radius
    }
},
n.Frustum = function(e, t, i, r, o, a) {
    this.planes = [void 0 !== e ? e : new n.Plane, void 0 !== t ? t : new n.Plane, void 0 !== i ? i : new n.Plane, void 0 !== r ? r : new n.Plane, void 0 !== o ? o : new n.Plane, void 0 !== a ? a : new n.Plane]
}
,
n.Frustum.prototype = {
    constructor: n.Frustum,
    set: function(e, t, i, n, r, o) {
        var a = this.planes;
        return a[0].copy(e),
        a[1].copy(t),
        a[2].copy(i),
        a[3].copy(n),
        a[4].copy(r),
        a[5].copy(o),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        for (var t = this.planes, i = 0; i < 6; i++)
            t[i].copy(e.planes[i]);
        return this
    },
    setFromMatrix: function(e) {
        var t = this.planes
          , i = e.elements
          , n = i[0]
          , r = i[1]
          , o = i[2]
          , a = i[3]
          , s = i[4]
          , l = i[5]
          , h = i[6]
          , c = i[7]
          , u = i[8]
          , d = i[9]
          , p = i[10]
          , f = i[11]
          , g = i[12]
          , m = i[13]
          , v = i[14]
          , A = i[15];
        return t[0].setComponents(a - n, c - s, f - u, A - g).normalize(),
        t[1].setComponents(a + n, c + s, f + u, A + g).normalize(),
        t[2].setComponents(a + r, c + l, f + d, A + m).normalize(),
        t[3].setComponents(a - r, c - l, f - d, A - m).normalize(),
        t[4].setComponents(a - o, c - h, f - p, A - v).normalize(),
        t[5].setComponents(a + o, c + h, f + p, A + v).normalize(),
        this
    },
    intersectsObject: function() {
        var e = new n.Sphere;
        return function(t) {
            var i = t.geometry;
            return null === i.boundingSphere && i.computeBoundingSphere(),
            e.copy(i.boundingSphere),
            e.applyMatrix4(t.matrixWorld),
            this.intersectsSphere(e)
        }
    }(),
    intersectsSphere: function(e) {
        for (var t = this.planes, i = e.center, n = -e.radius, r = 0; r < 6; r++) {
            var o = t[r].distanceToPoint(i);
            if (o < n)
                return !1
        }
        return !0
    },
    intersectsBox: function() {
        var e = new n.Vector3
          , t = new n.Vector3;
        return function(i) {
            for (var n = this.planes, r = 0; r < 6; r++) {
                var o = n[r];
                e.x = o.normal.x > 0 ? i.min.x : i.max.x,
                t.x = o.normal.x > 0 ? i.max.x : i.min.x,
                e.y = o.normal.y > 0 ? i.min.y : i.max.y,
                t.y = o.normal.y > 0 ? i.max.y : i.min.y,
                e.z = o.normal.z > 0 ? i.min.z : i.max.z,
                t.z = o.normal.z > 0 ? i.max.z : i.min.z;
                var a = o.distanceToPoint(e)
                  , s = o.distanceToPoint(t);
                if (a < 0 && s < 0)
                    return !1
            }
            return !0
        }
    }(),
    containsPoint: function(e) {
        for (var t = this.planes, i = 0; i < 6; i++)
            if (t[i].distanceToPoint(e) < 0)
                return !1;
        return !0
    }
},
n.Plane = function(e, t) {
    this.normal = void 0 !== e ? e : new n.Vector3(1,0,0),
    this.constant = void 0 !== t ? t : 0
}
,
n.Plane.prototype = {
    constructor: n.Plane,
    set: function(e, t) {
        return this.normal.copy(e),
        this.constant = t,
        this
    },
    setComponents: function(e, t, i, n) {
        return this.normal.set(e, t, i),
        this.constant = n,
        this
    },
    setFromNormalAndCoplanarPoint: function(e, t) {
        return this.normal.copy(e),
        this.constant = -t.dot(this.normal),
        this
    },
    setFromCoplanarPoints: function() {
        var e = new n.Vector3
          , t = new n.Vector3;
        return function(i, n, r) {
            var o = e.subVectors(r, n).cross(t.subVectors(i, n)).normalize();
            return this.setFromNormalAndCoplanarPoint(o, i),
            this
        }
    }(),
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.normal.copy(e.normal),
        this.constant = e.constant,
        this
    },
    normalize: function() {
        var e = 1 / this.normal.length();
        return this.normal.multiplyScalar(e),
        this.constant *= e,
        this
    },
    negate: function() {
        return this.constant *= -1,
        this.normal.negate(),
        this
    },
    distanceToPoint: function(e) {
        return this.normal.dot(e) + this.constant
    },
    distanceToSphere: function(e) {
        return this.distanceToPoint(e.center) - e.radius
    },
    projectPoint: function(e, t) {
        return this.orthoPoint(e, t).sub(e).negate()
    },
    orthoPoint: function(e, t) {
        var i = this.distanceToPoint(e)
          , r = t || new n.Vector3;
        return r.copy(this.normal).multiplyScalar(i)
    },
    intersectLine: function() {
        var e = new n.Vector3;
        return function(t, i) {
            var r = i || new n.Vector3
              , o = t.delta(e)
              , a = this.normal.dot(o);
            if (0 !== a) {
                var s = -(t.start.dot(this.normal) + this.constant) / a;
                if (!(s < 0 || s > 1))
                    return r.copy(o).multiplyScalar(s).add(t.start)
            } else if (0 === this.distanceToPoint(t.start))
                return r.copy(t.start)
        }
    }(),
    intersectsLine: function(e) {
        var t = this.distanceToPoint(e.start)
          , i = this.distanceToPoint(e.end);
        return t < 0 && i > 0 || i < 0 && t > 0
    },
    intersectsBox: function(e) {
        return e.intersectsPlane(this)
    },
    intersectsSphere: function(e) {
        return e.intersectsPlane(this)
    },
    coplanarPoint: function(e) {
        var t = e || new n.Vector3;
        return t.copy(this.normal).multiplyScalar(-this.constant)
    },
    applyMatrix4: function() {
        var e = new n.Vector3
          , t = new n.Vector3
          , i = new n.Matrix3;
        return function(n, r) {
            var o = r || i.getNormalMatrix(n)
              , a = e.copy(this.normal).applyMatrix3(o)
              , s = this.coplanarPoint(t);
            return s.applyMatrix4(n),
            this.setFromNormalAndCoplanarPoint(a, s),
            this
        }
    }(),
    translate: function(e) {
        return this.constant = this.constant - e.dot(this.normal),
        this
    },
    equals: function(e) {
        return e.normal.equals(this.normal) && e.constant === this.constant
    }
},
n.Spherical = function(e, t, i) {
    return this.radius = void 0 !== e ? e : 1,
    this.phi = void 0 !== t ? t : 0,
    this.theta = void 0 !== i ? i : 0,
    this
}
,
n.Spherical.prototype = {
    constructor: n.Spherical,
    set: function(e, t, i) {
        this.radius = e,
        this.phi = t,
        this.theta = i
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.radius.copy(e.radius),
        this.phi.copy(e.phi),
        this.theta.copy(e.theta),
        this
    },
    makeSafe: function() {
        var e = 1e-6;
        this.phi = Math.max(e, Math.min(Math.PI - e, this.phi))
    },
    setFromVector3: function(e) {
        return this.radius = e.length(),
        0 === this.radius ? (this.theta = 0,
        this.phi = 0) : (this.theta = Math.atan2(e.x, e.z),
        this.phi = Math.acos(n.Math.clamp(e.y / this.radius, -1, 1))),
        this
    }
},
n.Math = {
    generateUUID: function() {
        var e, t = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".split(""), i = new Array(36), n = 0;
        return function() {
            for (var r = 0; r < 36; r++)
                8 === r || 13 === r || 18 === r || 23 === r ? i[r] = "-" : 14 === r ? i[r] = "4" : (n <= 2 && (n = 33554432 + 16777216 * Math.random() | 0),
                e = 15 & n,
                n >>= 4,
                i[r] = t[19 === r ? 3 & e | 8 : e]);
            return i.join("")
        }
    }(),
    clamp: function(e, t, i) {
        return Math.max(t, Math.min(i, e))
    },
    euclideanModulo: function(e, t) {
        return (e % t + t) % t
    },
    mapLinear: function(e, t, i, n, r) {
        return n + (e - t) * (r - n) / (i - t)
    },
    smoothstep: function(e, t, i) {
        return e <= t ? 0 : e >= i ? 1 : (e = (e - t) / (i - t),
        e * e * (3 - 2 * e))
    },
    smootherstep: function(e, t, i) {
        return e <= t ? 0 : e >= i ? 1 : (e = (e - t) / (i - t),
        e * e * e * (e * (6 * e - 15) + 10))
    },
    random16: function() {
        return console.warn("THREE.Math.random16() has been deprecated. Use Math.random() instead."),
        Math.random()
    },
    randInt: function(e, t) {
        return e + Math.floor(Math.random() * (t - e + 1))
    },
    randFloat: function(e, t) {
        return e + Math.random() * (t - e)
    },
    randFloatSpread: function(e) {
        return e * (.5 - Math.random())
    },
    degToRad: function() {
        var e = Math.PI / 180;
        return function(t) {
            return t * e
        }
    }(),
    radToDeg: function() {
        var e = 180 / Math.PI;
        return function(t) {
            return t * e
        }
    }(),
    isPowerOfTwo: function(e) {
        return 0 === (e & e - 1) && 0 !== e
    },
    nearestPowerOfTwo: function(e) {
        return Math.pow(2, Math.round(Math.log(e) / Math.LN2))
    },
    nextPowerOfTwo: function(e) {
        return e--,
        e |= e >> 1,
        e |= e >> 2,
        e |= e >> 4,
        e |= e >> 8,
        e |= e >> 16,
        e++,
        e
    }
},
n.Spline = function(e) {
    function t(e, t, i, n, r, o, a) {
        var s = .5 * (i - e)
          , l = .5 * (n - t);
        return (2 * (t - i) + s + l) * a + (-3 * (t - i) - 2 * s - l) * o + s * r + t
    }
    this.points = e;
    var i, r, o, a, s, l, h, c, u, d = [], p = {
        x: 0,
        y: 0,
        z: 0
    };
    this.initFromArray = function(e) {
        this.points = [];
        for (var t = 0; t < e.length; t++)
            this.points[t] = {
                x: e[t][0],
                y: e[t][1],
                z: e[t][2]
            }
    }
    ,
    this.getPoint = function(e) {
        return i = (this.points.length - 1) * e,
        r = Math.floor(i),
        o = i - r,
        d[0] = 0 === r ? r : r - 1,
        d[1] = r,
        d[2] = r > this.points.length - 2 ? this.points.length - 1 : r + 1,
        d[3] = r > this.points.length - 3 ? this.points.length - 1 : r + 2,
        l = this.points[d[0]],
        h = this.points[d[1]],
        c = this.points[d[2]],
        u = this.points[d[3]],
        a = o * o,
        s = o * a,
        p.x = t(l.x, h.x, c.x, u.x, o, a, s),
        p.y = t(l.y, h.y, c.y, u.y, o, a, s),
        p.z = t(l.z, h.z, c.z, u.z, o, a, s),
        p
    }
    ,
    this.getControlPointsArray = function() {
        var e, t, i = this.points.length, n = [];
        for (e = 0; e < i; e++)
            t = this.points[e],
            n[e] = [t.x, t.y, t.z];
        return n
    }
    ,
    this.getLength = function(e) {
        var t, i, r, o, a = 0, s = 0, l = 0, h = new n.Vector3, c = new n.Vector3, u = [], d = 0;
        for (u[0] = 0,
        e || (e = 100),
        r = this.points.length * e,
        h.copy(this.points[0]),
        t = 1; t < r; t++)
            i = t / r,
            o = this.getPoint(i),
            c.copy(o),
            d += c.distanceTo(h),
            h.copy(o),
            a = (this.points.length - 1) * i,
            s = Math.floor(a),
            s !== l && (u[s] = d,
            l = s);
        return u[u.length] = d,
        {
            chunks: u,
            total: d
        }
    }
    ,
    this.reparametrizeByArcLength = function(e) {
        var t, i, r, o, a, s, l, h, c = [], u = new n.Vector3, d = this.getLength();
        for (c.push(u.copy(this.points[0]).clone()),
        t = 1; t < this.points.length; t++) {
            for (s = d.chunks[t] - d.chunks[t - 1],
            l = Math.ceil(e * s / d.total),
            o = (t - 1) / (this.points.length - 1),
            a = t / (this.points.length - 1),
            i = 1; i < l - 1; i++)
                r = o + i * (1 / l) * (a - o),
                h = this.getPoint(r),
                c.push(u.copy(h).clone());
            c.push(u.copy(this.points[t]).clone())
        }
        this.points = c
    }
}
,
n.Triangle = function(e, t, i) {
    this.a = void 0 !== e ? e : new n.Vector3,
    this.b = void 0 !== t ? t : new n.Vector3,
    this.c = void 0 !== i ? i : new n.Vector3
}
,
n.Triangle.normal = function() {
    var e = new n.Vector3;
    return function(t, i, r, o) {
        var a = o || new n.Vector3;
        a.subVectors(r, i),
        e.subVectors(t, i),
        a.cross(e);
        var s = a.lengthSq();
        return s > 0 ? a.multiplyScalar(1 / Math.sqrt(s)) : a.set(0, 0, 0)
    }
}(),
n.Triangle.barycoordFromPoint = function() {
    var e = new n.Vector3
      , t = new n.Vector3
      , i = new n.Vector3;
    return function(r, o, a, s, l) {
        e.subVectors(s, o),
        t.subVectors(a, o),
        i.subVectors(r, o);
        var h = e.dot(e)
          , c = e.dot(t)
          , u = e.dot(i)
          , d = t.dot(t)
          , p = t.dot(i)
          , f = h * d - c * c
          , g = l || new n.Vector3;
        if (0 === f)
            return g.set(-2, -1, -1);
        var m = 1 / f
          , v = (d * u - c * p) * m
          , A = (h * p - c * u) * m;
        return g.set(1 - v - A, A, v)
    }
}(),
n.Triangle.containsPoint = function() {
    var e = new n.Vector3;
    return function(t, i, r, o) {
        var a = n.Triangle.barycoordFromPoint(t, i, r, o, e);
        return a.x >= 0 && a.y >= 0 && a.x + a.y <= 1
    }
}(),
n.Triangle.prototype = {
    constructor: n.Triangle,
    set: function(e, t, i) {
        return this.a.copy(e),
        this.b.copy(t),
        this.c.copy(i),
        this
    },
    setFromPointsAndIndices: function(e, t, i, n) {
        return this.a.copy(e[t]),
        this.b.copy(e[i]),
        this.c.copy(e[n]),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.a.copy(e.a),
        this.b.copy(e.b),
        this.c.copy(e.c),
        this
    },
    area: function() {
        var e = new n.Vector3
          , t = new n.Vector3;
        return function() {
            return e.subVectors(this.c, this.b),
            t.subVectors(this.a, this.b),
            .5 * e.cross(t).length()
        }
    }(),
    midpoint: function(e) {
        var t = e || new n.Vector3;
        return t.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3)
    },
    normal: function(e) {
        return n.Triangle.normal(this.a, this.b, this.c, e)
    },
    plane: function(e) {
        var t = e || new n.Plane;
        return t.setFromCoplanarPoints(this.a, this.b, this.c)
    },
    barycoordFromPoint: function(e, t) {
        return n.Triangle.barycoordFromPoint(e, this.a, this.b, this.c, t)
    },
    containsPoint: function(e) {
        return n.Triangle.containsPoint(e, this.a, this.b, this.c)
    },
    equals: function(e) {
        return e.a.equals(this.a) && e.b.equals(this.b) && e.c.equals(this.c)
    }
},
n.Interpolant = function(e, t, i, n) {
    this.parameterPositions = e,
    this._cachedIndex = 0,
    this.resultBuffer = void 0 !== n ? n : new t.constructor(i),
    this.sampleValues = t,
    this.valueSize = i
}
,
n.Interpolant.prototype = {
    constructor: n.Interpolant,
    evaluate: function(e) {
        var t = this.parameterPositions
          , i = this._cachedIndex
          , n = t[i]
          , r = t[i - 1];
        e: {
            t: {
                var o;
                i: {
                    n: if (!(e < n)) {
                        for (var a = i + 2; ; ) {
                            if (void 0 === n) {
                                if (e < r)
                                    break n;
                                return i = t.length,
                                this._cachedIndex = i,
                                this.afterEnd_(i - 1, e, r)
                            }
                            if (i === a)
                                break;
                            if (r = n,
                            n = t[++i],
                            e < n)
                                break t
                        }
                        o = t.length;
                        break i
                    }
                    {
                        if (e >= r)
                            break e;
                        var s = t[1];
                        e < s && (i = 2,
                        r = s);
                        for (var a = i - 2; ; ) {
                            if (void 0 === r)
                                return this._cachedIndex = 0,
                                this.beforeStart_(0, e, n);
                            if (i === a)
                                break;
                            if (n = r,
                            r = t[--i - 1],
                            e >= r)
                                break t
                        }
                        o = i,
                        i = 0
                    }
                }
                for (; i < o; ) {
                    var l = i + o >>> 1;
                    e < t[l] ? o = l : i = l + 1
                }
                if (n = t[i],
                r = t[i - 1],
                void 0 === r)
                    return this._cachedIndex = 0,
                    this.beforeStart_(0, e, n);
                if (void 0 === n)
                    return i = t.length,
                    this._cachedIndex = i,
                    this.afterEnd_(i - 1, r, e)
            }
            this._cachedIndex = i,
            this.intervalChanged_(i, r, n)
        }
        return this.interpolate_(i, r, e, n)
    },
    settings: null,
    DefaultSettings_: {},
    getSettings_: function() {
        return this.settings || this.DefaultSettings_
    },
    copySampleValue_: function(e) {
        for (var t = this.resultBuffer, i = this.sampleValues, n = this.valueSize, r = e * n, o = 0; o !== n; ++o)
            t[o] = i[r + o];
        return t
    },
    interpolate_: function(e, t, i, n) {
        throw new Error("call to abstract method")
    },
    intervalChanged_: function(e, t, i) {}
},
Object.assign(n.Interpolant.prototype, {
    beforeStart_: n.Interpolant.prototype.copySampleValue_,
    afterEnd_: n.Interpolant.prototype.copySampleValue_
}),
n.CubicInterpolant = function(e, t, i, r) {
    n.Interpolant.call(this, e, t, i, r),
    this._weightPrev = -0,
    this._offsetPrev = -0,
    this._weightNext = -0,
    this._offsetNext = -0
}
,
n.CubicInterpolant.prototype = Object.assign(Object.create(n.Interpolant.prototype), {
    constructor: n.CubicInterpolant,
    DefaultSettings_: {
        endingStart: n.ZeroCurvatureEnding,
        endingEnd: n.ZeroCurvatureEnding
    },
    intervalChanged_: function(e, t, i) {
        var r = this.parameterPositions
          , o = e - 2
          , a = e + 1
          , s = r[o]
          , l = r[a];
        if (void 0 === s)
            switch (this.getSettings_().endingStart) {
            case n.ZeroSlopeEnding:
                o = e,
                s = 2 * t - i;
                break;
            case n.WrapAroundEnding:
                o = r.length - 2,
                s = t + r[o] - r[o + 1];
                break;
            default:
                o = e,
                s = i
            }
        if (void 0 === l)
            switch (this.getSettings_().endingEnd) {
            case n.ZeroSlopeEnding:
                a = e,
                l = 2 * i - t;
                break;
            case n.WrapAroundEnding:
                a = 1,
                l = i + r[1] - r[0];
                break;
            default:
                a = e - 1,
                l = t
            }
        var h = .5 * (i - t)
          , c = this.valueSize;
        this._weightPrev = h / (t - s),
        this._weightNext = h / (l - i),
        this._offsetPrev = o * c,
        this._offsetNext = a * c
    },
    interpolate_: function(e, t, i, n) {
        for (var r = this.resultBuffer, o = this.sampleValues, a = this.valueSize, s = e * a, l = s - a, h = this._offsetPrev, c = this._offsetNext, u = this._weightPrev, d = this._weightNext, p = (i - t) / (n - t), f = p * p, g = f * p, m = -u * g + 2 * u * f - u * p, v = (1 + u) * g + (-1.5 - 2 * u) * f + (-.5 + u) * p + 1, A = (-1 - d) * g + (1.5 + d) * f + .5 * p, y = d * g - d * f, C = 0; C !== a; ++C)
            r[C] = m * o[h + C] + v * o[l + C] + A * o[s + C] + y * o[c + C];
        return r
    }
}),
n.DiscreteInterpolant = function(e, t, i, r) {
    n.Interpolant.call(this, e, t, i, r)
}
,
n.DiscreteInterpolant.prototype = Object.assign(Object.create(n.Interpolant.prototype), {
    constructor: n.DiscreteInterpolant,
    interpolate_: function(e, t, i, n) {
        return this.copySampleValue_(e - 1)
    }
}),
n.LinearInterpolant = function(e, t, i, r) {
    n.Interpolant.call(this, e, t, i, r)
}
,
n.LinearInterpolant.prototype = Object.assign(Object.create(n.Interpolant.prototype), {
    constructor: n.LinearInterpolant,
    interpolate_: function(e, t, i, n) {
        for (var r = this.resultBuffer, o = this.sampleValues, a = this.valueSize, s = e * a, l = s - a, h = (i - t) / (n - t), c = 1 - h, u = 0; u !== a; ++u)
            r[u] = o[l + u] * c + o[s + u] * h;
        return r
    }
}),
n.QuaternionLinearInterpolant = function(e, t, i, r) {
    n.Interpolant.call(this, e, t, i, r)
}
,
n.QuaternionLinearInterpolant.prototype = Object.assign(Object.create(n.Interpolant.prototype), {
    constructor: n.QuaternionLinearInterpolant,
    interpolate_: function(e, t, i, r) {
        for (var o = this.resultBuffer, a = this.sampleValues, s = this.valueSize, l = e * s, h = (i - t) / (r - t), c = l + s; l !== c; l += 4)
            n.Quaternion.slerpFlat(o, 0, a, l - s, a, l, h);
        return o
    }
}),
n.Clock = function(e) {
    this.autoStart = void 0 === e || e,
    this.startTime = 0,
    this.oldTime = 0,
    this.elapsedTime = 0,
    this.running = !1
}
,
n.Clock.prototype = {
    constructor: n.Clock,
    start: function() {
        this.startTime = performance.now(),
        this.oldTime = this.startTime,
        this.running = !0
    },
    stop: function() {
        this.getElapsedTime(),
        this.running = !1
    },
    getElapsedTime: function() {
        return this.getDelta(),
        this.elapsedTime
    },
    getDelta: function() {
        var e = 0;
        if (this.autoStart && !this.running && this.start(),
        this.running) {
            var t = performance.now();
            e = .001 * (t - this.oldTime),
            this.oldTime = t,
            this.elapsedTime += e
        }
        return e
    }
},
n.EventDispatcher = function() {}
,
n.EventDispatcher.prototype = {
    constructor: n.EventDispatcher,
    apply: function(e) {
        e.addEventListener = n.EventDispatcher.prototype.addEventListener,
        e.hasEventListener = n.EventDispatcher.prototype.hasEventListener,
        e.removeEventListener = n.EventDispatcher.prototype.removeEventListener,
        e.dispatchEvent = n.EventDispatcher.prototype.dispatchEvent
    },
    addEventListener: function(e, t) {
        void 0 === this._listeners && (this._listeners = {});
        var i = this._listeners;
        void 0 === i[e] && (i[e] = []),
        i[e].indexOf(t) === -1 && i[e].push(t)
    },
    hasEventListener: function(e, t) {
        if (void 0 === this._listeners)
            return !1;
        var i = this._listeners;
        return void 0 !== i[e] && i[e].indexOf(t) !== -1
    },
    removeEventListener: function(e, t) {
        if (void 0 !== this._listeners) {
            var i = this._listeners
              , n = i[e];
            if (void 0 !== n) {
                var r = n.indexOf(t);
                r !== -1 && n.splice(r, 1)
            }
        }
    },
    dispatchEvent: function(e) {
        if (void 0 !== this._listeners) {
            var t = this._listeners
              , i = t[e.type];
            if (void 0 !== i) {
                e.target = this;
                for (var n = [], r = i.length, o = 0; o < r; o++)
                    n[o] = i[o];
                for (var o = 0; o < r; o++)
                    n[o].call(this, e)
            }
        }
    }
},
n.Layers = function() {
    this.mask = 1
}
,
n.Layers.prototype = {
    constructor: n.Layers,
    set: function(e) {
        this.mask = 1 << e
    },
    enable: function(e) {
        this.mask |= 1 << e
    },
    toggle: function(e) {
        this.mask ^= 1 << e
    },
    disable: function(e) {
        this.mask &= ~(1 << e)
    },
    test: function(e) {
        return 0 !== (this.mask & e.mask)
    }
},
function(e) {
    function t(e, t) {
        return e.distance - t.distance
    }
    function i(e, t, n, r) {
        if (e.visible !== !1 && (e.raycast(t, n),
        r === !0))
            for (var o = e.children, a = 0, s = o.length; a < s; a++)
                i(o[a], t, n, !0)
    }
    e.Raycaster = function(t, i, n, r) {
        this.ray = new e.Ray(t,i),
        this.near = n || 0,
        this.far = r || 1 / 0,
        this.params = {
            Mesh: {},
            Line: {},
            LOD: {},
            Points: {
                threshold: 1
            },
            Sprite: {}
        },
        Object.defineProperties(this.params, {
            PointCloud: {
                get: function() {
                    return console.warn("THREE.Raycaster: params.PointCloud has been renamed to params.Points."),
                    this.Points
                }
            }
        })
    }
    ,
    e.Raycaster.prototype = {
        constructor: e.Raycaster,
        linePrecision: 1,
        set: function(e, t) {
            this.ray.set(e, t)
        },
        setFromCamera: function(t, i) {
            i instanceof e.PerspectiveCamera ? (this.ray.origin.setFromMatrixPosition(i.matrixWorld),
            this.ray.direction.set(t.x, t.y, .5).unproject(i).sub(this.ray.origin).normalize()) : i instanceof e.OrthographicCamera ? (this.ray.origin.set(t.x, t.y, -1).unproject(i),
            this.ray.direction.set(0, 0, -1).transformDirection(i.matrixWorld)) : console.error("THREE.Raycaster: Unsupported camera type.")
        },
        intersectObject: function(e, n) {
            var r = [];
            return i(e, this, r, n),
            r.sort(t),
            r
        },
        intersectObjects: function(e, n) {
            var r = [];
            if (Array.isArray(e) === !1)
                return console.warn("THREE.Raycaster.intersectObjects: objects is not an Array."),
                r;
            for (var o = 0, a = e.length; o < a; o++)
                i(e[o], this, r, n);
            return r.sort(t),
            r
        }
    }
}(n),
n.Object3D = function() {
    function e() {
        o.setFromEuler(r, !1)
    }
    function t() {
        r.setFromQuaternion(o, void 0, !1)
    }
    Object.defineProperty(this, "id", {
        value: n.Object3DIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.type = "Object3D",
    this.parent = null,
    this.children = [],
    this.up = n.Object3D.DefaultUp.clone();
    var i = new n.Vector3
      , r = new n.Euler
      , o = new n.Quaternion
      , a = new n.Vector3(1,1,1);
    r.onChange(e),
    o.onChange(t),
    Object.defineProperties(this, {
        position: {
            enumerable: !0,
            value: i
        },
        rotation: {
            enumerable: !0,
            value: r
        },
        quaternion: {
            enumerable: !0,
            value: o
        },
        scale: {
            enumerable: !0,
            value: a
        },
        modelViewMatrix: {
            value: new n.Matrix4
        },
        normalMatrix: {
            value: new n.Matrix3
        }
    }),
    this.rotationAutoUpdate = !0,
    this.matrix = new n.Matrix4,
    this.matrixWorld = new n.Matrix4,
    this.matrixAutoUpdate = n.Object3D.DefaultMatrixAutoUpdate,
    this.matrixWorldNeedsUpdate = !1,
    this.layers = new n.Layers,
    this.visible = !0,
    this.castShadow = !1,
    this.receiveShadow = !1,
    this.frustumCulled = !0,
    this.renderOrder = 0,
    this.userData = {}
}
,
n.Object3D.DefaultUp = new n.Vector3(0,1,0),
n.Object3D.DefaultMatrixAutoUpdate = !0,
n.Object3D.prototype = {
    constructor: n.Object3D,
    applyMatrix: function(e) {
        this.matrix.multiplyMatrices(e, this.matrix),
        this.matrix.decompose(this.position, this.quaternion, this.scale)
    },
    setRotationFromAxisAngle: function(e, t) {
        this.quaternion.setFromAxisAngle(e, t)
    },
    setRotationFromEuler: function(e) {
        this.quaternion.setFromEuler(e, !0)
    },
    setRotationFromMatrix: function(e) {
        this.quaternion.setFromRotationMatrix(e)
    },
    setRotationFromQuaternion: function(e) {
        this.quaternion.copy(e)
    },
    rotateOnAxis: function() {
        var e = new n.Quaternion;
        return function(t, i) {
            return e.setFromAxisAngle(t, i),
            this.quaternion.multiply(e),
            this
        }
    }(),
    rotateX: function() {
        var e = new n.Vector3(1,0,0);
        return function(t) {
            return this.rotateOnAxis(e, t)
        }
    }(),
    rotateY: function() {
        var e = new n.Vector3(0,1,0);
        return function(t) {
            return this.rotateOnAxis(e, t)
        }
    }(),
    rotateZ: function() {
        var e = new n.Vector3(0,0,1);
        return function(t) {
            return this.rotateOnAxis(e, t)
        }
    }(),
    translateOnAxis: function() {
        var e = new n.Vector3;
        return function(t, i) {
            return e.copy(t).applyQuaternion(this.quaternion),
            this.position.add(e.multiplyScalar(i)),
            this
        }
    }(),
    translateX: function() {
        var e = new n.Vector3(1,0,0);
        return function(t) {
            return this.translateOnAxis(e, t)
        }
    }(),
    translateY: function() {
        var e = new n.Vector3(0,1,0);
        return function(t) {
            return this.translateOnAxis(e, t)
        }
    }(),
    translateZ: function() {
        var e = new n.Vector3(0,0,1);
        return function(t) {
            return this.translateOnAxis(e, t)
        }
    }(),
    localToWorld: function(e) {
        return e.applyMatrix4(this.matrixWorld)
    },
    worldToLocal: function() {
        var e = new n.Matrix4;
        return function(t) {
            return t.applyMatrix4(e.getInverse(this.matrixWorld))
        }
    }(),
    lookAt: function() {
        var e = new n.Matrix4;
        return function(t) {
            e.lookAt(t, this.position, this.up),
            this.quaternion.setFromRotationMatrix(e)
        }
    }(),
    add: function(e) {
        if (arguments.length > 1) {
            for (var t = 0; t < arguments.length; t++)
                this.add(arguments[t]);
            return this
        }
        return e === this ? (console.error("THREE.Object3D.add: object can't be added as a child of itself.", e),
        this) : (e instanceof n.Object3D ? (null !== e.parent && e.parent.remove(e),
        e.parent = this,
        e.dispatchEvent({
            type: "added"
        }),
        this.children.push(e)) : console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.", e),
        this)
    },
    remove: function(e) {
        if (arguments.length > 1)
            for (var t = 0; t < arguments.length; t++)
                this.remove(arguments[t]);
        var i = this.children.indexOf(e);
        i !== -1 && (e.parent = null,
        e.dispatchEvent({
            type: "removed"
        }),
        this.children.splice(i, 1))
    },
    getObjectById: function(e) {
        return this.getObjectByProperty("id", e)
    },
    getObjectByName: function(e) {
        return this.getObjectByProperty("name", e)
    },
    getObjectByProperty: function(e, t) {
        if (this[e] === t)
            return this;
        for (var i = 0, n = this.children.length; i < n; i++) {
            var r = this.children[i]
              , o = r.getObjectByProperty(e, t);
            if (void 0 !== o)
                return o
        }
    },
    getWorldPosition: function(e) {
        var t = e || new n.Vector3;
        return this.updateMatrixWorld(!0),
        t.setFromMatrixPosition(this.matrixWorld)
    },
    getWorldQuaternion: function() {
        var e = new n.Vector3
          , t = new n.Vector3;
        return function(i) {
            var r = i || new n.Quaternion;
            return this.updateMatrixWorld(!0),
            this.matrixWorld.decompose(e, r, t),
            r
        }
    }(),
    getWorldRotation: function() {
        var e = new n.Quaternion;
        return function(t) {
            var i = t || new n.Euler;
            return this.getWorldQuaternion(e),
            i.setFromQuaternion(e, this.rotation.order, !1)
        }
    }(),
    getWorldScale: function() {
        var e = new n.Vector3
          , t = new n.Quaternion;
        return function(i) {
            var r = i || new n.Vector3;
            return this.updateMatrixWorld(!0),
            this.matrixWorld.decompose(e, t, r),
            r
        }
    }(),
    getWorldDirection: function() {
        var e = new n.Quaternion;
        return function(t) {
            var i = t || new n.Vector3;
            return this.getWorldQuaternion(e),
            i.set(0, 0, 1).applyQuaternion(e)
        }
    }(),
    raycast: function() {},
    traverse: function(e) {
        e(this);
        for (var t = this.children, i = 0, n = t.length; i < n; i++)
            t[i].traverse(e)
    },
    traverseVisible: function(e) {
        if (this.visible !== !1) {
            e(this);
            for (var t = this.children, i = 0, n = t.length; i < n; i++)
                t[i].traverseVisible(e)
        }
    },
    traverseAncestors: function(e) {
        var t = this.parent;
        null !== t && (e(t),
        t.traverseAncestors(e))
    },
    updateMatrix: function() {
        this.matrix.compose(this.position, this.quaternion, this.scale),
        this.matrixWorldNeedsUpdate = !0
    },
    updateMatrixWorld: function(e) {
        this.matrixAutoUpdate === !0 && this.updateMatrix(),
        this.matrixWorldNeedsUpdate !== !0 && e !== !0 || (null === this.parent ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix),
        this.matrixWorldNeedsUpdate = !1,
        e = !0);
        for (var t = 0, i = this.children.length; t < i; t++)
            this.children[t].updateMatrixWorld(e)
    },
    toJSON: function(e) {
        function t(e) {
            var t = [];
            for (var i in e) {
                var n = e[i];
                delete n.metadata,
                t.push(n)
            }
            return t
        }
        var i = void 0 === e
          , n = {};
        i && (e = {
            geometries: {},
            materials: {},
            textures: {},
            images: {}
        },
        n.metadata = {
            version: 4.4,
            type: "Object",
            generator: "Object3D.toJSON"
        });
        var r = {};
        if (r.uuid = this.uuid,
        r.type = this.type,
        "" !== this.name && (r.name = this.name),
        "{}" !== JSON.stringify(this.userData) && (r.userData = this.userData),
        this.castShadow === !0 && (r.castShadow = !0),
        this.receiveShadow === !0 && (r.receiveShadow = !0),
        this.visible === !1 && (r.visible = !1),
        r.matrix = this.matrix.toArray(),
        void 0 !== this.geometry && (void 0 === e.geometries[this.geometry.uuid] && (e.geometries[this.geometry.uuid] = this.geometry.toJSON(e)),
        r.geometry = this.geometry.uuid),
        void 0 !== this.material && (void 0 === e.materials[this.material.uuid] && (e.materials[this.material.uuid] = this.material.toJSON(e)),
        r.material = this.material.uuid),
        this.children.length > 0) {
            r.children = [];
            for (var o = 0; o < this.children.length; o++)
                r.children.push(this.children[o].toJSON(e).object)
        }
        if (i) {
            var a = t(e.geometries)
              , s = t(e.materials)
              , l = t(e.textures)
              , h = t(e.images);
            a.length > 0 && (n.geometries = a),
            s.length > 0 && (n.materials = s),
            l.length > 0 && (n.textures = l),
            h.length > 0 && (n.images = h)
        }
        return n.object = r,
        n
    },
    clone: function(e) {
        return (new this.constructor).copy(this, e)
    },
    copy: function(e, t) {
        if (void 0 === t && (t = !0),
        this.name = e.name,
        this.up.copy(e.up),
        this.position.copy(e.position),
        this.quaternion.copy(e.quaternion),
        this.scale.copy(e.scale),
        this.rotationAutoUpdate = e.rotationAutoUpdate,
        this.matrix.copy(e.matrix),
        this.matrixWorld.copy(e.matrixWorld),
        this.matrixAutoUpdate = e.matrixAutoUpdate,
        this.matrixWorldNeedsUpdate = e.matrixWorldNeedsUpdate,
        this.visible = e.visible,
        this.castShadow = e.castShadow,
        this.receiveShadow = e.receiveShadow,
        this.frustumCulled = e.frustumCulled,
        this.renderOrder = e.renderOrder,
        this.userData = JSON.parse(JSON.stringify(e.userData)),
        t === !0)
            for (var i = 0; i < e.children.length; i++) {
                var n = e.children[i];
                this.add(n.clone())
            }
        return this
    }
},
n.EventDispatcher.prototype.apply(n.Object3D.prototype),
n.Object3DIdCount = 0,
n.Face3 = function(e, t, i, r, o, a) {
    this.a = e,
    this.b = t,
    this.c = i,
    this.normal = r instanceof n.Vector3 ? r : new n.Vector3,
    this.vertexNormals = Array.isArray(r) ? r : [],
    this.color = o instanceof n.Color ? o : new n.Color,
    this.vertexColors = Array.isArray(o) ? o : [],
    this.materialIndex = void 0 !== a ? a : 0
}
,
n.Face3.prototype = {
    constructor: n.Face3,
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        this.a = e.a,
        this.b = e.b,
        this.c = e.c,
        this.normal.copy(e.normal),
        this.color.copy(e.color),
        this.materialIndex = e.materialIndex;
        for (var t = 0, i = e.vertexNormals.length; t < i; t++)
            this.vertexNormals[t] = e.vertexNormals[t].clone();
        for (var t = 0, i = e.vertexColors.length; t < i; t++)
            this.vertexColors[t] = e.vertexColors[t].clone();
        return this
    }
},
n.BufferAttribute = function(e, t) {
    this.uuid = n.Math.generateUUID(),
    this.array = e,
    this.itemSize = t,
    this.dynamic = !1,
    this.updateRange = {
        offset: 0,
        count: -1
    },
    this.version = 0
}
,
n.BufferAttribute.prototype = {
    constructor: n.BufferAttribute,
    get count() {
        return this.array.length / this.itemSize
    },
    set needsUpdate(e) {
        e === !0 && this.version++
    },
    setDynamic: function(e) {
        return this.dynamic = e,
        this
    },
    copy: function(e) {
        return this.array = new e.array.constructor(e.array),
        this.itemSize = e.itemSize,
        this.dynamic = e.dynamic,
        this
    },
    copyAt: function(e, t, i) {
        e *= this.itemSize,
        i *= t.itemSize;
        for (var n = 0, r = this.itemSize; n < r; n++)
            this.array[e + n] = t.array[i + n];
        return this
    },
    copyArray: function(e) {
        return this.array.set(e),
        this
    },
    copyColorsArray: function(e) {
        for (var t = this.array, i = 0, r = 0, o = e.length; r < o; r++) {
            var a = e[r];
            void 0 === a && (console.warn("THREE.BufferAttribute.copyColorsArray(): color is undefined", r),
            a = new n.Color),
            t[i++] = a.r,
            t[i++] = a.g,
            t[i++] = a.b
        }
        return this
    },
    copyIndicesArray: function(e) {
        for (var t = this.array, i = 0, n = 0, r = e.length; n < r; n++) {
            var o = e[n];
            t[i++] = o.a,
            t[i++] = o.b,
            t[i++] = o.c
        }
        return this
    },
    copyVector2sArray: function(e) {
        for (var t = this.array, i = 0, r = 0, o = e.length; r < o; r++) {
            var a = e[r];
            void 0 === a && (console.warn("THREE.BufferAttribute.copyVector2sArray(): vector is undefined", r),
            a = new n.Vector2),
            t[i++] = a.x,
            t[i++] = a.y
        }
        return this
    },
    copyVector3sArray: function(e) {
        for (var t = this.array, i = 0, r = 0, o = e.length; r < o; r++) {
            var a = e[r];
            void 0 === a && (console.warn("THREE.BufferAttribute.copyVector3sArray(): vector is undefined", r),
            a = new n.Vector3),
            t[i++] = a.x,
            t[i++] = a.y,
            t[i++] = a.z
        }
        return this
    },
    copyVector4sArray: function(e) {
        for (var t = this.array, i = 0, r = 0, o = e.length; r < o; r++) {
            var a = e[r];
            void 0 === a && (console.warn("THREE.BufferAttribute.copyVector4sArray(): vector is undefined", r),
            a = new n.Vector4),
            t[i++] = a.x,
            t[i++] = a.y,
            t[i++] = a.z,
            t[i++] = a.w
        }
        return this
    },
    set: function(e, t) {
        return void 0 === t && (t = 0),
        this.array.set(e, t),
        this
    },
    getX: function(e) {
        return this.array[e * this.itemSize]
    },
    setX: function(e, t) {
        return this.array[e * this.itemSize] = t,
        this
    },
    getY: function(e) {
        return this.array[e * this.itemSize + 1]
    },
    setY: function(e, t) {
        return this.array[e * this.itemSize + 1] = t,
        this
    },
    getZ: function(e) {
        return this.array[e * this.itemSize + 2]
    },
    setZ: function(e, t) {
        return this.array[e * this.itemSize + 2] = t,
        this
    },
    getW: function(e) {
        return this.array[e * this.itemSize + 3]
    },
    setW: function(e, t) {
        return this.array[e * this.itemSize + 3] = t,
        this
    },
    setXY: function(e, t, i) {
        return e *= this.itemSize,
        this.array[e + 0] = t,
        this.array[e + 1] = i,
        this
    },
    setXYZ: function(e, t, i, n) {
        return e *= this.itemSize,
        this.array[e + 0] = t,
        this.array[e + 1] = i,
        this.array[e + 2] = n,
        this
    },
    setXYZW: function(e, t, i, n, r) {
        return e *= this.itemSize,
        this.array[e + 0] = t,
        this.array[e + 1] = i,
        this.array[e + 2] = n,
        this.array[e + 3] = r,
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    }
},
n.Int8Attribute = function(e, t) {
    return new n.BufferAttribute(new Int8Array(e),t)
}
,
n.Uint8Attribute = function(e, t) {
    return new n.BufferAttribute(new Uint8Array(e),t)
}
,
n.Uint8ClampedAttribute = function(e, t) {
    return new n.BufferAttribute(new Uint8ClampedArray(e),t)
}
,
n.Int16Attribute = function(e, t) {
    return new n.BufferAttribute(new Int16Array(e),t)
}
,
n.Uint16Attribute = function(e, t) {
    return new n.BufferAttribute(new Uint16Array(e),t)
}
;
n.Int32Attribute = function(e, t) {
    return new n.BufferAttribute(new Int32Array(e),t)
}
;
n.Uint32Attribute = function(e, t) {
    return new n.BufferAttribute(new Uint32Array(e),t)
}
,
n.Float32Attribute = function(e, t) {
    return new n.BufferAttribute(new Float32Array(e),t)
}
,
n.Float64Attribute = function(e, t) {
    return new n.BufferAttribute(new Float64Array(e),t)
}
,
n.DynamicBufferAttribute = function(e, t) {
    return console.warn("THREE.DynamicBufferAttribute has been removed. Use new THREE.BufferAttribute().setDynamic( true ) instead."),
    new n.BufferAttribute(e,t).setDynamic(!0)
}
,
n.InstancedBufferAttribute = function(e, t, i) {
    n.BufferAttribute.call(this, e, t),
    this.meshPerAttribute = i || 1
}
,
n.InstancedBufferAttribute.prototype = Object.create(n.BufferAttribute.prototype),
n.InstancedBufferAttribute.prototype.constructor = n.InstancedBufferAttribute,
n.InstancedBufferAttribute.prototype.copy = function(e) {
    return n.BufferAttribute.prototype.copy.call(this, e),
    this.meshPerAttribute = e.meshPerAttribute,
    this
}
,
n.InterleavedBuffer = function(e, t) {
    this.uuid = n.Math.generateUUID(),
    this.array = e,
    this.stride = t,
    this.dynamic = !1,
    this.updateRange = {
        offset: 0,
        count: -1
    },
    this.version = 0
}
,
n.InterleavedBuffer.prototype = {
    constructor: n.InterleavedBuffer,
    get length() {
        return this.array.length
    },
    get count() {
        return this.array.length / this.stride
    },
    set needsUpdate(e) {
        e === !0 && this.version++
    },
    setDynamic: function(e) {
        return this.dynamic = e,
        this
    },
    copy: function(e) {
        return this.array = new e.array.constructor(e.array),
        this.stride = e.stride,
        this.dynamic = e.dynamic,
        this
    },
    copyAt: function(e, t, i) {
        e *= this.stride,
        i *= t.stride;
        for (var n = 0, r = this.stride; n < r; n++)
            this.array[e + n] = t.array[i + n];
        return this
    },
    set: function(e, t) {
        return void 0 === t && (t = 0),
        this.array.set(e, t),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    }
},
n.InstancedInterleavedBuffer = function(e, t, i) {
    n.InterleavedBuffer.call(this, e, t),
    this.meshPerAttribute = i || 1
}
,
n.InstancedInterleavedBuffer.prototype = Object.create(n.InterleavedBuffer.prototype),
n.InstancedInterleavedBuffer.prototype.constructor = n.InstancedInterleavedBuffer,
n.InstancedInterleavedBuffer.prototype.copy = function(e) {
    return n.InterleavedBuffer.prototype.copy.call(this, e),
    this.meshPerAttribute = e.meshPerAttribute,
    this
}
,
n.InterleavedBufferAttribute = function(e, t, i) {
    this.uuid = n.Math.generateUUID(),
    this.data = e,
    this.itemSize = t,
    this.offset = i
}
,
n.InterleavedBufferAttribute.prototype = {
    constructor: n.InterleavedBufferAttribute,
    get length() {
        return console.warn("THREE.BufferAttribute: .length has been deprecated. Please use .count."),
        this.array.length
    },
    get count() {
        return this.data.count
    },
    setX: function(e, t) {
        return this.data.array[e * this.data.stride + this.offset] = t,
        this
    },
    setY: function(e, t) {
        return this.data.array[e * this.data.stride + this.offset + 1] = t,
        this
    },
    setZ: function(e, t) {
        return this.data.array[e * this.data.stride + this.offset + 2] = t,
        this
    },
    setW: function(e, t) {
        return this.data.array[e * this.data.stride + this.offset + 3] = t,
        this
    },
    getX: function(e) {
        return this.data.array[e * this.data.stride + this.offset]
    },
    getY: function(e) {
        return this.data.array[e * this.data.stride + this.offset + 1]
    },
    getZ: function(e) {
        return this.data.array[e * this.data.stride + this.offset + 2]
    },
    getW: function(e) {
        return this.data.array[e * this.data.stride + this.offset + 3]
    },
    setXY: function(e, t, i) {
        return e = e * this.data.stride + this.offset,
        this.data.array[e + 0] = t,
        this.data.array[e + 1] = i,
        this
    },
    setXYZ: function(e, t, i, n) {
        return e = e * this.data.stride + this.offset,
        this.data.array[e + 0] = t,
        this.data.array[e + 1] = i,
        this.data.array[e + 2] = n,
        this
    },
    setXYZW: function(e, t, i, n, r) {
        return e = e * this.data.stride + this.offset,
        this.data.array[e + 0] = t,
        this.data.array[e + 1] = i,
        this.data.array[e + 2] = n,
        this.data.array[e + 3] = r,
        this
    }
},
n.Geometry = function() {
    Object.defineProperty(this, "id", {
        value: n.GeometryIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.type = "Geometry",
    this.vertices = [],
    this.colors = [],
    this.faces = [],
    this.faceVertexUvs = [[]],
    this.morphTargets = [],
    this.morphNormals = [],
    this.skinWeights = [],
    this.skinIndices = [],
    this.lineDistances = [],
    this.boundingBox = null,
    this.boundingSphere = null,
    this.verticesNeedUpdate = !1,
    this.elementsNeedUpdate = !1,
    this.uvsNeedUpdate = !1,
    this.normalsNeedUpdate = !1,
    this.colorsNeedUpdate = !1,
    this.lineDistancesNeedUpdate = !1,
    this.groupsNeedUpdate = !1
}
,
n.Geometry.prototype = {
    constructor: n.Geometry,
    applyMatrix: function(e) {
        for (var t = (new n.Matrix3).getNormalMatrix(e), i = 0, r = this.vertices.length; i < r; i++) {
            var o = this.vertices[i];
            o.applyMatrix4(e)
        }
        for (var i = 0, r = this.faces.length; i < r; i++) {
            var a = this.faces[i];
            a.normal.applyMatrix3(t).normalize();
            for (var s = 0, l = a.vertexNormals.length; s < l; s++)
                a.vertexNormals[s].applyMatrix3(t).normalize()
        }
        return null !== this.boundingBox && this.computeBoundingBox(),
        null !== this.boundingSphere && this.computeBoundingSphere(),
        this.verticesNeedUpdate = !0,
        this.normalsNeedUpdate = !0,
        this
    },
    rotateX: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationX(t),
            this.applyMatrix(e),
            this
        }
    }(),
    rotateY: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationY(t),
            this.applyMatrix(e),
            this
        }
    }(),
    rotateZ: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationZ(t),
            this.applyMatrix(e),
            this
        }
    }(),
    translate: function() {
        var e;
        return function(t, i, r) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeTranslation(t, i, r),
            this.applyMatrix(e),
            this
        }
    }(),
    scale: function() {
        var e;
        return function(t, i, r) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeScale(t, i, r),
            this.applyMatrix(e),
            this
        }
    }(),
    lookAt: function() {
        var e;
        return function(t) {
            void 0 === e && (e = new n.Object3D),
            e.lookAt(t),
            e.updateMatrix(),
            this.applyMatrix(e.matrix)
        }
    }(),
    fromBufferGeometry: function(e) {
        function t(e, t, r, o) {
            var a = void 0 !== s ? [u[e].clone(), u[t].clone(), u[r].clone()] : []
              , f = void 0 !== l ? [i.colors[e].clone(), i.colors[t].clone(), i.colors[r].clone()] : []
              , g = new n.Face3(e,t,r,a,f,o);
            i.faces.push(g),
            void 0 !== h && i.faceVertexUvs[0].push([d[e].clone(), d[t].clone(), d[r].clone()]),
            void 0 !== c && i.faceVertexUvs[1].push([p[e].clone(), p[t].clone(), p[r].clone()])
        }
        var i = this
          , r = null !== e.index ? e.index.array : void 0
          , o = e.attributes
          , a = o.position.array
          , s = void 0 !== o.normal ? o.normal.array : void 0
          , l = void 0 !== o.color ? o.color.array : void 0
          , h = void 0 !== o.uv ? o.uv.array : void 0
          , c = void 0 !== o.uv2 ? o.uv2.array : void 0;
        void 0 !== c && (this.faceVertexUvs[1] = []);
        for (var u = [], d = [], p = [], f = 0, g = 0; f < a.length; f += 3,
        g += 2)
            i.vertices.push(new n.Vector3(a[f],a[f + 1],a[f + 2])),
            void 0 !== s && u.push(new n.Vector3(s[f],s[f + 1],s[f + 2])),
            void 0 !== l && i.colors.push(new n.Color(l[f],l[f + 1],l[f + 2])),
            void 0 !== h && d.push(new n.Vector2(h[g],h[g + 1])),
            void 0 !== c && p.push(new n.Vector2(c[g],c[g + 1]));
        if (void 0 !== r) {
            var m = e.groups;
            if (m.length > 0)
                for (var f = 0; f < m.length; f++)
                    for (var v = m[f], A = v.start, y = v.count, g = A, C = A + y; g < C; g += 3)
                        t(r[g], r[g + 1], r[g + 2], v.materialIndex);
            else
                for (var f = 0; f < r.length; f += 3)
                    t(r[f], r[f + 1], r[f + 2])
        } else
            for (var f = 0; f < a.length / 3; f += 3)
                t(f, f + 1, f + 2);
        return this.computeFaceNormals(),
        null !== e.boundingBox && (this.boundingBox = e.boundingBox.clone()),
        null !== e.boundingSphere && (this.boundingSphere = e.boundingSphere.clone()),
        this
    },
    center: function() {
        this.computeBoundingBox();
        var e = this.boundingBox.center().negate();
        return this.translate(e.x, e.y, e.z),
        e
    },
    normalize: function() {
        this.computeBoundingSphere();
        var e = this.boundingSphere.center
          , t = this.boundingSphere.radius
          , i = 0 === t ? 1 : 1 / t
          , r = new n.Matrix4;
        return r.set(i, 0, 0, -i * e.x, 0, i, 0, -i * e.y, 0, 0, i, -i * e.z, 0, 0, 0, 1),
        this.applyMatrix(r),
        this
    },
    computeFaceNormals: function() {
        for (var e = new n.Vector3, t = new n.Vector3, i = 0, r = this.faces.length; i < r; i++) {
            var o = this.faces[i]
              , a = this.vertices[o.a]
              , s = this.vertices[o.b]
              , l = this.vertices[o.c];
            e.subVectors(l, s),
            t.subVectors(a, s),
            e.cross(t),
            e.normalize(),
            o.normal.copy(e)
        }
    },
    computeVertexNormals: function(e) {
        void 0 === e && (e = !0);
        var t, i, r, o, a, s;
        for (s = new Array(this.vertices.length),
        t = 0,
        i = this.vertices.length; t < i; t++)
            s[t] = new n.Vector3;
        if (e) {
            var l, h, c, u = new n.Vector3, d = new n.Vector3;
            for (r = 0,
            o = this.faces.length; r < o; r++)
                a = this.faces[r],
                l = this.vertices[a.a],
                h = this.vertices[a.b],
                c = this.vertices[a.c],
                u.subVectors(c, h),
                d.subVectors(l, h),
                u.cross(d),
                s[a.a].add(u),
                s[a.b].add(u),
                s[a.c].add(u)
        } else
            for (r = 0,
            o = this.faces.length; r < o; r++)
                a = this.faces[r],
                s[a.a].add(a.normal),
                s[a.b].add(a.normal),
                s[a.c].add(a.normal);
        for (t = 0,
        i = this.vertices.length; t < i; t++)
            s[t].normalize();
        for (r = 0,
        o = this.faces.length; r < o; r++) {
            a = this.faces[r];
            var p = a.vertexNormals;
            3 === p.length ? (p[0].copy(s[a.a]),
            p[1].copy(s[a.b]),
            p[2].copy(s[a.c])) : (p[0] = s[a.a].clone(),
            p[1] = s[a.b].clone(),
            p[2] = s[a.c].clone())
        }
        this.faces.length > 0 && (this.normalsNeedUpdate = !0)
    },
    computeMorphNormals: function() {
        var e, t, i, r, o;
        for (i = 0,
        r = this.faces.length; i < r; i++)
            for (o = this.faces[i],
            o.__originalFaceNormal ? o.__originalFaceNormal.copy(o.normal) : o.__originalFaceNormal = o.normal.clone(),
            o.__originalVertexNormals || (o.__originalVertexNormals = []),
            e = 0,
            t = o.vertexNormals.length; e < t; e++)
                o.__originalVertexNormals[e] ? o.__originalVertexNormals[e].copy(o.vertexNormals[e]) : o.__originalVertexNormals[e] = o.vertexNormals[e].clone();
        var a = new n.Geometry;
        for (a.faces = this.faces,
        e = 0,
        t = this.morphTargets.length; e < t; e++) {
            if (!this.morphNormals[e]) {
                this.morphNormals[e] = {},
                this.morphNormals[e].faceNormals = [],
                this.morphNormals[e].vertexNormals = [];
                var s, l, h = this.morphNormals[e].faceNormals, c = this.morphNormals[e].vertexNormals;
                for (i = 0,
                r = this.faces.length; i < r; i++)
                    s = new n.Vector3,
                    l = {
                        a: new n.Vector3,
                        b: new n.Vector3,
                        c: new n.Vector3
                    },
                    h.push(s),
                    c.push(l)
            }
            var u = this.morphNormals[e];
            a.vertices = this.morphTargets[e].vertices,
            a.computeFaceNormals(),
            a.computeVertexNormals();
            var s, l;
            for (i = 0,
            r = this.faces.length; i < r; i++)
                o = this.faces[i],
                s = u.faceNormals[i],
                l = u.vertexNormals[i],
                s.copy(o.normal),
                l.a.copy(o.vertexNormals[0]),
                l.b.copy(o.vertexNormals[1]),
                l.c.copy(o.vertexNormals[2])
        }
        for (i = 0,
        r = this.faces.length; i < r; i++)
            o = this.faces[i],
            o.normal = o.__originalFaceNormal,
            o.vertexNormals = o.__originalVertexNormals
    },
    computeTangents: function() {
        console.warn("THREE.Geometry: .computeTangents() has been removed.")
    },
    computeLineDistances: function() {
        for (var e = 0, t = this.vertices, i = 0, n = t.length; i < n; i++)
            i > 0 && (e += t[i].distanceTo(t[i - 1])),
            this.lineDistances[i] = e
    },
    computeBoundingBox: function() {
        null === this.boundingBox && (this.boundingBox = new n.Box3),
        this.boundingBox.setFromPoints(this.vertices)
    },
    computeBoundingSphere: function() {
        null === this.boundingSphere && (this.boundingSphere = new n.Sphere),
        this.boundingSphere.setFromPoints(this.vertices)
    },
    merge: function(e, t, i) {
        if (e instanceof n.Geometry == !1)
            return void console.error("THREE.Geometry.merge(): geometry not an instance of THREE.Geometry.", e);
        var r, o = this.vertices.length, a = this.vertices, s = e.vertices, l = this.faces, h = e.faces, c = this.faceVertexUvs[0], u = e.faceVertexUvs[0];
        void 0 === i && (i = 0),
        void 0 !== t && (r = (new n.Matrix3).getNormalMatrix(t));
        for (var d = 0, p = s.length; d < p; d++) {
            var f = s[d]
              , g = f.clone();
            void 0 !== t && g.applyMatrix4(t),
            a.push(g)
        }
        for (d = 0,
        p = h.length; d < p; d++) {
            var m, v, A, y = h[d], C = y.vertexNormals, I = y.vertexColors;
            m = new n.Face3(y.a + o,y.b + o,y.c + o),
            m.normal.copy(y.normal),
            void 0 !== r && m.normal.applyMatrix3(r).normalize();
            for (var b = 0, w = C.length; b < w; b++)
                v = C[b].clone(),
                void 0 !== r && v.applyMatrix3(r).normalize(),
                m.vertexNormals.push(v);
            m.color.copy(y.color);
            for (var b = 0, w = I.length; b < w; b++)
                A = I[b],
                m.vertexColors.push(A.clone());
            m.materialIndex = y.materialIndex + i,
            l.push(m)
        }
        for (d = 0,
        p = u.length; d < p; d++) {
            var E = u[d]
              , x = [];
            if (void 0 !== E) {
                for (var b = 0, w = E.length; b < w; b++)
                    x.push(E[b].clone());
                c.push(x)
            }
        }
    },
    mergeMesh: function(e) {
        return e instanceof n.Mesh == !1 ? void console.error("THREE.Geometry.mergeMesh(): mesh not an instance of THREE.Mesh.", e) : (e.matrixAutoUpdate && e.updateMatrix(),
        void this.merge(e.geometry, e.matrix))
    },
    mergeVertices: function() {
        var e, t, i, n, r, o, a, s, l = {}, h = [], c = [], u = 4, d = Math.pow(10, u);
        for (i = 0,
        n = this.vertices.length; i < n; i++)
            e = this.vertices[i],
            t = Math.round(e.x * d) + "_" + Math.round(e.y * d) + "_" + Math.round(e.z * d),
            void 0 === l[t] ? (l[t] = i,
            h.push(this.vertices[i]),
            c[i] = h.length - 1) : c[i] = c[l[t]];
        var p = [];
        for (i = 0,
        n = this.faces.length; i < n; i++) {
            r = this.faces[i],
            r.a = c[r.a],
            r.b = c[r.b],
            r.c = c[r.c],
            o = [r.a, r.b, r.c];
            for (var f = -1, g = 0; g < 3; g++)
                if (o[g] === o[(g + 1) % 3]) {
                    f = g,
                    p.push(i);
                    break
                }
        }
        for (i = p.length - 1; i >= 0; i--) {
            var m = p[i];
            for (this.faces.splice(m, 1),
            a = 0,
            s = this.faceVertexUvs.length; a < s; a++)
                this.faceVertexUvs[a].splice(m, 1)
        }
        var v = this.vertices.length - h.length;
        return this.vertices = h,
        v
    },
    sortFacesByMaterialIndex: function() {
        function e(e, t) {
            return e.materialIndex - t.materialIndex
        }
        for (var t = this.faces, i = t.length, n = 0; n < i; n++)
            t[n]._id = n;
        t.sort(e);
        var r, o, a = this.faceVertexUvs[0], s = this.faceVertexUvs[1];
        a && a.length === i && (r = []),
        s && s.length === i && (o = []);
        for (var n = 0; n < i; n++) {
            var l = t[n]._id;
            r && r.push(a[l]),
            o && o.push(s[l])
        }
        r && (this.faceVertexUvs[0] = r),
        o && (this.faceVertexUvs[1] = o)
    },
    toJSON: function() {
        function e(e, t, i) {
            return i ? e | 1 << t : e & ~(1 << t)
        }
        function t(e) {
            var t = e.x.toString() + e.y.toString() + e.z.toString();
            return void 0 !== d[t] ? d[t] : (d[t] = u.length / 3,
            u.push(e.x, e.y, e.z),
            d[t])
        }
        function i(e) {
            var t = e.r.toString() + e.g.toString() + e.b.toString();
            return void 0 !== f[t] ? f[t] : (f[t] = p.length,
            p.push(e.getHex()),
            f[t])
        }
        function n(e) {
            var t = e.x.toString() + e.y.toString();
            return void 0 !== m[t] ? m[t] : (m[t] = g.length / 2,
            g.push(e.x, e.y),
            m[t])
        }
        var r = {
            metadata: {
                version: 4.4,
                type: "Geometry",
                generator: "Geometry.toJSON"
            }
        };
        if (r.uuid = this.uuid,
        r.type = this.type,
        "" !== this.name && (r.name = this.name),
        void 0 !== this.parameters) {
            var o = this.parameters;
            for (var a in o)
                void 0 !== o[a] && (r[a] = o[a]);
            return r
        }
        for (var s = [], l = 0; l < this.vertices.length; l++) {
            var h = this.vertices[l];
            s.push(h.x, h.y, h.z)
        }
        for (var c = [], u = [], d = {}, p = [], f = {}, g = [], m = {}, l = 0; l < this.faces.length; l++) {
            var v = this.faces[l]
              , A = !0
              , y = !1
              , C = void 0 !== this.faceVertexUvs[0][l]
              , I = v.normal.length() > 0
              , b = v.vertexNormals.length > 0
              , w = 1 !== v.color.r || 1 !== v.color.g || 1 !== v.color.b
              , E = v.vertexColors.length > 0
              , x = 0;
            if (x = e(x, 0, 0),
            x = e(x, 1, A),
            x = e(x, 2, y),
            x = e(x, 3, C),
            x = e(x, 4, I),
            x = e(x, 5, b),
            x = e(x, 6, w),
            x = e(x, 7, E),
            c.push(x),
            c.push(v.a, v.b, v.c),
            c.push(v.materialIndex),
            C) {
                var T = this.faceVertexUvs[0][l];
                c.push(n(T[0]), n(T[1]), n(T[2]))
            }
            if (I && c.push(t(v.normal)),
            b) {
                var M = v.vertexNormals;
                c.push(t(M[0]), t(M[1]), t(M[2]))
            }
            if (w && c.push(i(v.color)),
            E) {
                var S = v.vertexColors;
                c.push(i(S[0]), i(S[1]), i(S[2]))
            }
        }
        return r.data = {},
        r.data.vertices = s,
        r.data.normals = u,
        p.length > 0 && (r.data.colors = p),
        g.length > 0 && (r.data.uvs = [g]),
        r.data.faces = c,
        r
    },
    clone: function() {
        return (new n.Geometry).copy(this)
    },
    copy: function(e) {
        this.vertices = [],
        this.faces = [],
        this.faceVertexUvs = [[]];
        for (var t = e.vertices, i = 0, n = t.length; i < n; i++)
            this.vertices.push(t[i].clone());
        for (var r = e.faces, i = 0, n = r.length; i < n; i++)
            this.faces.push(r[i].clone());
        for (var i = 0, n = e.faceVertexUvs.length; i < n; i++) {
            var o = e.faceVertexUvs[i];
            void 0 === this.faceVertexUvs[i] && (this.faceVertexUvs[i] = []);
            for (var a = 0, s = o.length; a < s; a++) {
                for (var l = o[a], h = [], c = 0, u = l.length; c < u; c++) {
                    var d = l[c];
                    h.push(d.clone())
                }
                this.faceVertexUvs[i].push(h)
            }
        }
        return this
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    }
},
n.EventDispatcher.prototype.apply(n.Geometry.prototype),
n.GeometryIdCount = 0,
n.DirectGeometry = function() {
    Object.defineProperty(this, "id", {
        value: n.GeometryIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.type = "DirectGeometry",
    this.indices = [],
    this.vertices = [],
    this.normals = [],
    this.colors = [],
    this.uvs = [],
    this.uvs2 = [],
    this.groups = [],
    this.morphTargets = {},
    this.skinWeights = [],
    this.skinIndices = [],
    this.boundingBox = null,
    this.boundingSphere = null,
    this.verticesNeedUpdate = !1,
    this.normalsNeedUpdate = !1,
    this.colorsNeedUpdate = !1,
    this.uvsNeedUpdate = !1,
    this.groupsNeedUpdate = !1
}
,
n.DirectGeometry.prototype = {
    constructor: n.DirectGeometry,
    computeBoundingBox: n.Geometry.prototype.computeBoundingBox,
    computeBoundingSphere: n.Geometry.prototype.computeBoundingSphere,
    computeFaceNormals: function() {
        console.warn("THREE.DirectGeometry: computeFaceNormals() is not a method of this type of geometry.")
    },
    computeVertexNormals: function() {
        console.warn("THREE.DirectGeometry: computeVertexNormals() is not a method of this type of geometry.")
    },
    computeGroups: function(e) {
        for (var t, i, n = [], r = e.faces, o = 0; o < r.length; o++) {
            var a = r[o];
            a.materialIndex !== i && (i = a.materialIndex,
            void 0 !== t && (t.count = 3 * o - t.start,
            n.push(t)),
            t = {
                start: 3 * o,
                materialIndex: i
            })
        }
        void 0 !== t && (t.count = 3 * o - t.start,
        n.push(t)),
        this.groups = n
    },
    fromGeometry: function(e) {
        var t, i = e.faces, r = e.vertices, o = e.faceVertexUvs, a = o[0] && o[0].length > 0, s = o[1] && o[1].length > 0, l = e.morphTargets, h = l.length;
        if (h > 0) {
            t = [];
            for (var c = 0; c < h; c++)
                t[c] = [];
            this.morphTargets.position = t
        }
        var u, d = e.morphNormals, p = d.length;
        if (p > 0) {
            u = [];
            for (var c = 0; c < p; c++)
                u[c] = [];
            this.morphTargets.normal = u
        }
        for (var f = e.skinIndices, g = e.skinWeights, m = f.length === r.length, v = g.length === r.length, c = 0; c < i.length; c++) {
            var A = i[c];
            this.vertices.push(r[A.a], r[A.b], r[A.c]);
            var y = A.vertexNormals;
            if (3 === y.length)
                this.normals.push(y[0], y[1], y[2]);
            else {
                var C = A.normal;
                this.normals.push(C, C, C)
            }
            var I = A.vertexColors;
            if (3 === I.length)
                this.colors.push(I[0], I[1], I[2]);
            else {
                var b = A.color;
                this.colors.push(b, b, b)
            }
            if (a === !0) {
                var w = o[0][c];
                void 0 !== w ? this.uvs.push(w[0], w[1], w[2]) : (console.warn("THREE.DirectGeometry.fromGeometry(): Undefined vertexUv ", c),
                this.uvs.push(new n.Vector2, new n.Vector2, new n.Vector2))
            }
            if (s === !0) {
                var w = o[1][c];
                void 0 !== w ? this.uvs2.push(w[0], w[1], w[2]) : (console.warn("THREE.DirectGeometry.fromGeometry(): Undefined vertexUv2 ", c),
                this.uvs2.push(new n.Vector2, new n.Vector2, new n.Vector2))
            }
            for (var E = 0; E < h; E++) {
                var x = l[E].vertices;
                t[E].push(x[A.a], x[A.b], x[A.c])
            }
            for (var E = 0; E < p; E++) {
                var T = d[E].vertexNormals[c];
                u[E].push(T.a, T.b, T.c)
            }
            m && this.skinIndices.push(f[A.a], f[A.b], f[A.c]),
            v && this.skinWeights.push(g[A.a], g[A.b], g[A.c])
        }
        return this.computeGroups(e),
        this.verticesNeedUpdate = e.verticesNeedUpdate,
        this.normalsNeedUpdate = e.normalsNeedUpdate,
        this.colorsNeedUpdate = e.colorsNeedUpdate,
        this.uvsNeedUpdate = e.uvsNeedUpdate,
        this.groupsNeedUpdate = e.groupsNeedUpdate,
        this
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    }
},
n.EventDispatcher.prototype.apply(n.DirectGeometry.prototype),
n.BufferGeometry = function() {
    Object.defineProperty(this, "id", {
        value: n.GeometryIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.type = "BufferGeometry",
    this.index = null,
    this.attributes = {},
    this.morphAttributes = {},
    this.groups = [],
    this.boundingBox = null,
    this.boundingSphere = null,
    this.drawRange = {
        start: 0,
        count: 1 / 0
    }
}
,
n.BufferGeometry.prototype = {
    constructor: n.BufferGeometry,
    getIndex: function() {
        return this.index
    },
    setIndex: function(e) {
        this.index = e
    },
    addAttribute: function(e, t) {
        return t instanceof n.BufferAttribute == !1 && t instanceof n.InterleavedBufferAttribute == !1 ? (console.warn("THREE.BufferGeometry: .addAttribute() now expects ( name, attribute )."),
        void this.addAttribute(e, new n.BufferAttribute(arguments[1],arguments[2]))) : "index" === e ? (console.warn("THREE.BufferGeometry.addAttribute: Use .setIndex() for index attribute."),
        void this.setIndex(t)) : (this.attributes[e] = t,
        this)
    },
    getAttribute: function(e) {
        return this.attributes[e]
    },
    removeAttribute: function(e) {
        return delete this.attributes[e],
        this
    },
    addGroup: function(e, t, i) {
        this.groups.push({
            start: e,
            count: t,
            materialIndex: void 0 !== i ? i : 0
        })
    },
    clearGroups: function() {
        this.groups = []
    },
    setDrawRange: function(e, t) {
        this.drawRange.start = e,
        this.drawRange.count = t
    },
    applyMatrix: function(e) {
        var t = this.attributes.position;
        void 0 !== t && (e.applyToVector3Array(t.array),
        t.needsUpdate = !0);
        var i = this.attributes.normal;
        if (void 0 !== i) {
            var r = (new n.Matrix3).getNormalMatrix(e);
            r.applyToVector3Array(i.array),
            i.needsUpdate = !0
        }
        return null !== this.boundingBox && this.computeBoundingBox(),
        null !== this.boundingSphere && this.computeBoundingSphere(),
        this
    },
    rotateX: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationX(t),
            this.applyMatrix(e),
            this
        }
    }(),
    rotateY: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationY(t),
            this.applyMatrix(e),
            this
        }
    }(),
    rotateZ: function() {
        var e;
        return function(t) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeRotationZ(t),
            this.applyMatrix(e),
            this
        }
    }(),
    translate: function() {
        var e;
        return function(t, i, r) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeTranslation(t, i, r),
            this.applyMatrix(e),
            this
        }
    }(),
    scale: function() {
        var e;
        return function(t, i, r) {
            return void 0 === e && (e = new n.Matrix4),
            e.makeScale(t, i, r),
            this.applyMatrix(e),
            this
        }
    }(),
    lookAt: function() {
        var e;
        return function(t) {
            void 0 === e && (e = new n.Object3D),
            e.lookAt(t),
            e.updateMatrix(),
            this.applyMatrix(e.matrix)
        }
    }(),
    center: function() {
        this.computeBoundingBox();
        var e = this.boundingBox.center().negate();
        return this.translate(e.x, e.y, e.z),
        e
    },
    setFromObject: function(e) {
        var t = e.geometry;
        if (e instanceof n.Points || e instanceof n.Line) {
            var i = new n.Float32Attribute(3 * t.vertices.length,3)
              , r = new n.Float32Attribute(3 * t.colors.length,3);
            if (this.addAttribute("position", i.copyVector3sArray(t.vertices)),
            this.addAttribute("color", r.copyColorsArray(t.colors)),
            t.lineDistances && t.lineDistances.length === t.vertices.length) {
                var o = new n.Float32Attribute(t.lineDistances.length,1);
                this.addAttribute("lineDistance", o.copyArray(t.lineDistances))
            }
            null !== t.boundingSphere && (this.boundingSphere = t.boundingSphere.clone()),
            null !== t.boundingBox && (this.boundingBox = t.boundingBox.clone())
        } else
            e instanceof n.Mesh && t instanceof n.Geometry && this.fromGeometry(t);
        return this
    },
    updateFromObject: function(e) {
        var t = e.geometry;
        if (e instanceof n.Mesh) {
            var i = t.__directGeometry;
            if (void 0 === i)
                return this.fromGeometry(t);
            i.verticesNeedUpdate = t.verticesNeedUpdate,
            i.normalsNeedUpdate = t.normalsNeedUpdate,
            i.colorsNeedUpdate = t.colorsNeedUpdate,
            i.uvsNeedUpdate = t.uvsNeedUpdate,
            i.groupsNeedUpdate = t.groupsNeedUpdate,
            t.verticesNeedUpdate = !1,
            t.normalsNeedUpdate = !1,
            t.colorsNeedUpdate = !1,
            t.uvsNeedUpdate = !1,
            t.groupsNeedUpdate = !1,
            t = i
        }
        if (t.verticesNeedUpdate === !0) {
            var r = this.attributes.position;
            void 0 !== r && (r.copyVector3sArray(t.vertices),
            r.needsUpdate = !0),
            t.verticesNeedUpdate = !1
        }
        if (t.normalsNeedUpdate === !0) {
            var r = this.attributes.normal;
            void 0 !== r && (r.copyVector3sArray(t.normals),
            r.needsUpdate = !0),
            t.normalsNeedUpdate = !1
        }
        if (t.colorsNeedUpdate === !0) {
            var r = this.attributes.color;
            void 0 !== r && (r.copyColorsArray(t.colors),
            r.needsUpdate = !0),
            t.colorsNeedUpdate = !1
        }
        if (t.uvsNeedUpdate) {
            var r = this.attributes.uv;
            void 0 !== r && (r.copyVector2sArray(t.uvs),
            r.needsUpdate = !0),
            t.uvsNeedUpdate = !1
        }
        if (t.lineDistancesNeedUpdate) {
            var r = this.attributes.lineDistance;
            void 0 !== r && (r.copyArray(t.lineDistances),
            r.needsUpdate = !0),
            t.lineDistancesNeedUpdate = !1
        }
        return t.groupsNeedUpdate && (t.computeGroups(e.geometry),
        this.groups = t.groups,
        t.groupsNeedUpdate = !1),
        this
    },
    fromGeometry: function(e) {
        return e.__directGeometry = (new n.DirectGeometry).fromGeometry(e),
        this.fromDirectGeometry(e.__directGeometry)
    },
    fromDirectGeometry: function(e) {
        var t = new Float32Array(3 * e.vertices.length);
        if (this.addAttribute("position", new n.BufferAttribute(t,3).copyVector3sArray(e.vertices)),
        e.normals.length > 0) {
            var i = new Float32Array(3 * e.normals.length);
            this.addAttribute("normal", new n.BufferAttribute(i,3).copyVector3sArray(e.normals))
        }
        if (e.colors.length > 0) {
            var r = new Float32Array(3 * e.colors.length);
            this.addAttribute("color", new n.BufferAttribute(r,3).copyColorsArray(e.colors))
        }
        if (e.uvs.length > 0) {
            var o = new Float32Array(2 * e.uvs.length);
            this.addAttribute("uv", new n.BufferAttribute(o,2).copyVector2sArray(e.uvs))
        }
        if (e.uvs2.length > 0) {
            var a = new Float32Array(2 * e.uvs2.length);
            this.addAttribute("uv2", new n.BufferAttribute(a,2).copyVector2sArray(e.uvs2))
        }
        if (e.indices.length > 0) {
            var s = e.vertices.length > 65535 ? Uint32Array : Uint16Array
              , l = new s(3 * e.indices.length);
            this.setIndex(new n.BufferAttribute(l,1).copyIndicesArray(e.indices))
        }
        this.groups = e.groups;
        for (var h in e.morphTargets) {
            for (var c = [], u = e.morphTargets[h], d = 0, p = u.length; d < p; d++) {
                var f = u[d]
                  , g = new n.Float32Attribute(3 * f.length,3);
                c.push(g.copyVector3sArray(f))
            }
            this.morphAttributes[h] = c
        }
        if (e.skinIndices.length > 0) {
            var m = new n.Float32Attribute(4 * e.skinIndices.length,4);
            this.addAttribute("skinIndex", m.copyVector4sArray(e.skinIndices))
        }
        if (e.skinWeights.length > 0) {
            var v = new n.Float32Attribute(4 * e.skinWeights.length,4);
            this.addAttribute("skinWeight", v.copyVector4sArray(e.skinWeights))
        }
        return null !== e.boundingSphere && (this.boundingSphere = e.boundingSphere.clone()),
        null !== e.boundingBox && (this.boundingBox = e.boundingBox.clone()),
        this
    },
    computeBoundingBox: function() {
        new n.Vector3;
        return function() {
            null === this.boundingBox && (this.boundingBox = new n.Box3);
            var e = this.attributes.position.array;
            e && this.boundingBox.setFromArray(e),
            void 0 !== e && 0 !== e.length || (this.boundingBox.min.set(0, 0, 0),
            this.boundingBox.max.set(0, 0, 0)),
            (isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && console.error('THREE.BufferGeometry.computeBoundingBox: Computed min/max have NaN values. The "position" attribute is likely to have NaN values.', this)
        }
    }(),
    computeBoundingSphere: function() {
        var e = new n.Box3
          , t = new n.Vector3;
        return function() {
            null === this.boundingSphere && (this.boundingSphere = new n.Sphere);
            var i = this.attributes.position.array;
            if (i) {
                var r = this.boundingSphere.center;
                e.setFromArray(i),
                e.center(r);
                for (var o = 0, a = 0, s = i.length; a < s; a += 3)
                    t.fromArray(i, a),
                    o = Math.max(o, r.distanceToSquared(t));
                this.boundingSphere.radius = Math.sqrt(o),
                isNaN(this.boundingSphere.radius) && console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.', this)
            }
        }
    }(),
    computeFaceNormals: function() {},
    computeVertexNormals: function() {
        var e = this.index
          , t = this.attributes
          , i = this.groups;
        if (t.position) {
            var r = t.position.array;
            if (void 0 === t.normal)
                this.addAttribute("normal", new n.BufferAttribute(new Float32Array(r.length),3));
            else
                for (var o = t.normal.array, a = 0, s = o.length; a < s; a++)
                    o[a] = 0;
            var l, h, c, u = t.normal.array, d = new n.Vector3, p = new n.Vector3, f = new n.Vector3, g = new n.Vector3, m = new n.Vector3;
            if (e) {
                var v = e.array;
                0 === i.length && this.addGroup(0, v.length);
                for (var A = 0, y = i.length; A < y; ++A)
                    for (var C = i[A], I = C.start, b = C.count, a = I, s = I + b; a < s; a += 3)
                        l = 3 * v[a + 0],
                        h = 3 * v[a + 1],
                        c = 3 * v[a + 2],
                        d.fromArray(r, l),
                        p.fromArray(r, h),
                        f.fromArray(r, c),
                        g.subVectors(f, p),
                        m.subVectors(d, p),
                        g.cross(m),
                        u[l] += g.x,
                        u[l + 1] += g.y,
                        u[l + 2] += g.z,
                        u[h] += g.x,
                        u[h + 1] += g.y,
                        u[h + 2] += g.z,
                        u[c] += g.x,
                        u[c + 1] += g.y,
                        u[c + 2] += g.z
            } else
                for (var a = 0, s = r.length; a < s; a += 9)
                    d.fromArray(r, a),
                    p.fromArray(r, a + 3),
                    f.fromArray(r, a + 6),
                    g.subVectors(f, p),
                    m.subVectors(d, p),
                    g.cross(m),
                    u[a] = g.x,
                    u[a + 1] = g.y,
                    u[a + 2] = g.z,
                    u[a + 3] = g.x,
                    u[a + 4] = g.y,
                    u[a + 5] = g.z,
                    u[a + 6] = g.x,
                    u[a + 7] = g.y,
                    u[a + 8] = g.z;
            this.normalizeNormals(),
            t.normal.needsUpdate = !0
        }
    },
    merge: function(e, t) {
        if (e instanceof n.BufferGeometry == !1)
            return void console.error("THREE.BufferGeometry.merge(): geometry not an instance of THREE.BufferGeometry.", e);
        void 0 === t && (t = 0);
        var i = this.attributes;
        for (var r in i)
            if (void 0 !== e.attributes[r])
                for (var o = i[r], a = o.array, s = e.attributes[r], l = s.array, h = s.itemSize, c = 0, u = h * t; c < l.length; c++,
                u++)
                    a[u] = l[c];
        return this
    },
    normalizeNormals: function() {
        for (var e, t, i, n, r = this.attributes.normal.array, o = 0, a = r.length; o < a; o += 3)
            e = r[o],
            t = r[o + 1],
            i = r[o + 2],
            n = 1 / Math.sqrt(e * e + t * t + i * i),
            r[o] *= n,
            r[o + 1] *= n,
            r[o + 2] *= n
    },
    toNonIndexed: function() {
        if (null === this.index)
            return console.warn("THREE.BufferGeometry.toNonIndexed(): Geometry is already non-indexed."),
            this;
        var e = new n.BufferGeometry
          , t = this.index.array
          , i = this.attributes;
        for (var r in i) {
            for (var o = i[r], a = o.array, s = o.itemSize, l = new a.constructor(t.length * s), h = 0, c = 0, u = 0, d = t.length; u < d; u++) {
                h = t[u] * s;
                for (var p = 0; p < s; p++)
                    l[c++] = a[h++]
            }
            e.addAttribute(r, new n.BufferAttribute(l,s))
        }
        return e
    },
    toJSON: function() {
        var e = {
            metadata: {
                version: 4.4,
                type: "BufferGeometry",
                generator: "BufferGeometry.toJSON"
            }
        };
        if (e.uuid = this.uuid,
        e.type = this.type,
        "" !== this.name && (e.name = this.name),
        void 0 !== this.parameters) {
            var t = this.parameters;
            for (var i in t)
                void 0 !== t[i] && (e[i] = t[i]);
            return e
        }
        e.data = {
            attributes: {}
        };
        var n = this.index;
        if (null !== n) {
            var r = Array.prototype.slice.call(n.array);
            e.data.index = {
                type: n.array.constructor.name,
                array: r
            }
        }
        var o = this.attributes;
        for (var i in o) {
            var a = o[i]
              , r = Array.prototype.slice.call(a.array);
            e.data.attributes[i] = {
                itemSize: a.itemSize,
                type: a.array.constructor.name,
                array: r
            }
        }
        var s = this.groups;
        s.length > 0 && (e.data.groups = JSON.parse(JSON.stringify(s)));
        var l = this.boundingSphere;
        return null !== l && (e.data.boundingSphere = {
            center: l.center.toArray(),
            radius: l.radius
        }),
        e
    },
    clone: function() {
        return (new n.BufferGeometry).copy(this)
    },
    copy: function(e) {
        var t = e.index;
        null !== t && this.setIndex(t.clone());
        var i = e.attributes;
        for (var n in i) {
            var r = i[n];
            this.addAttribute(n, r.clone())
        }
        for (var o = e.groups, a = 0, s = o.length; a < s; a++) {
            var l = o[a];
            this.addGroup(l.start, l.count)
        }
        return this
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    }
},
n.EventDispatcher.prototype.apply(n.BufferGeometry.prototype),
n.BufferGeometry.MaxIndex = 65535,
n.InstancedBufferGeometry = function() {
    n.BufferGeometry.call(this),
    this.type = "InstancedBufferGeometry",
    this.maxInstancedCount = void 0
}
,
n.InstancedBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.InstancedBufferGeometry.prototype.constructor = n.InstancedBufferGeometry,
n.InstancedBufferGeometry.prototype.addGroup = function(e, t, i) {
    this.groups.push({
        start: e,
        count: t,
        instances: i
    })
}
,
n.InstancedBufferGeometry.prototype.copy = function(e) {
    var t = e.index;
    null !== t && this.setIndex(t.clone());
    var i = e.attributes;
    for (var n in i) {
        var r = i[n];
        this.addAttribute(n, r.clone())
    }
    for (var o = e.groups, a = 0, s = o.length; a < s; a++) {
        var l = o[a];
        this.addGroup(l.start, l.count, l.instances)
    }
    return this
}
,
n.EventDispatcher.prototype.apply(n.InstancedBufferGeometry.prototype),
n.Uniform = function(e, t) {
    this.type = e,
    this.value = t,
    this.dynamic = !1
}
,
n.Uniform.prototype = {
    constructor: n.Uniform,
    onUpdate: function(e) {
        return this.dynamic = !0,
        this.onUpdateCallback = e,
        this
    }
},
n.AnimationClip = function(e, t, i) {
    this.name = e || n.Math.generateUUID(),
    this.tracks = i,
    this.duration = void 0 !== t ? t : -1,
    this.duration < 0 && this.resetDuration(),
    this.trim(),
    this.optimize()
}
,
n.AnimationClip.prototype = {
    constructor: n.AnimationClip,
    resetDuration: function() {
        for (var e = this.tracks, t = 0, i = 0, n = e.length; i !== n; ++i) {
            var r = this.tracks[i];
            t = Math.max(t, r.times[r.times.length - 1])
        }
        this.duration = t
    },
    trim: function() {
        for (var e = 0; e < this.tracks.length; e++)
            this.tracks[e].trim(0, this.duration);
        return this
    },
    optimize: function() {
        for (var e = 0; e < this.tracks.length; e++)
            this.tracks[e].optimize();
        return this
    }
},
Object.assign(n.AnimationClip, {
    parse: function(e) {
        for (var t = [], i = e.tracks, r = 1 / (e.fps || 1), o = 0, a = i.length; o !== a; ++o)
            t.push(n.KeyframeTrack.parse(i[o]).scale(r));
        return new n.AnimationClip(e.name,e.duration,t)
    },
    toJSON: function(e) {
        for (var t = [], i = e.tracks, r = {
            name: e.name,
            duration: e.duration,
            tracks: t
        }, o = 0, a = i.length; o !== a; ++o)
            t.push(n.KeyframeTrack.toJSON(i[o]));
        return r
    },
    CreateFromMorphTargetSequence: function(e, t, i) {
        for (var r = t.length, o = [], a = 0; a < r; a++) {
            var s = []
              , l = [];
            s.push((a + r - 1) % r, a, (a + 1) % r),
            l.push(0, 1, 0);
            var h = n.AnimationUtils.getKeyframeOrder(s);
            s = n.AnimationUtils.sortedArray(s, 1, h),
            l = n.AnimationUtils.sortedArray(l, 1, h),
            0 === s[0] && (s.push(r),
            l.push(l[0])),
            o.push(new n.NumberKeyframeTrack(".morphTargetInfluences[" + t[a].name + "]",s,l).scale(1 / i))
        }
        return new n.AnimationClip(e,-1,o)
    },
    findByName: function(e, t) {
        for (var i = 0; i < e.length; i++)
            if (e[i].name === t)
                return e[i];
        return null
    },
    CreateClipsFromMorphTargetSequences: function(e, t) {
        for (var i = {}, r = /^([\w-]*?)([\d]+)$/, o = 0, a = e.length; o < a; o++) {
            var s = e[o]
              , l = s.name.match(r);
            if (l && l.length > 1) {
                var h = l[1]
                  , c = i[h];
                c || (i[h] = c = []),
                c.push(s)
            }
        }
        var u = [];
        for (var h in i)
            u.push(n.AnimationClip.CreateFromMorphTargetSequence(h, i[h], t));
        return u
    },
    parseAnimation: function(e, t, i) {
        if (!e)
            return console.error("  no animation in JSONLoader data"),
            null;
        for (var r = function(e, t, i, r, o) {
            if (0 !== i.length) {
                var a = []
                  , s = [];
                n.AnimationUtils.flattenJSON(i, a, s, r),
                0 !== a.length && o.push(new e(t,a,s))
            }
        }, o = [], a = e.name || "default", s = e.length || -1, l = e.fps || 30, h = e.hierarchy || [], c = 0; c < h.length; c++) {
            var u = h[c].keys;
            if (u && 0 != u.length)
                if (u[0].morphTargets) {
                    for (var d = {}, p = 0; p < u.length; p++)
                        if (u[p].morphTargets)
                            for (var f = 0; f < u[p].morphTargets.length; f++)
                                d[u[p].morphTargets[f]] = -1;
                    for (var g in d) {
                        for (var m = [], v = [], f = 0; f !== u[p].morphTargets.length; ++f) {
                            var A = u[p];
                            m.push(A.time),
                            v.push(A.morphTarget === g ? 1 : 0)
                        }
                        o.push(new n.NumberKeyframeTrack(".morphTargetInfluence[" + g + "]",m,v))
                    }
                    s = d.length * (l || 1)
                } else {
                    var y = ".bones[" + t[c].name + "]";
                    r(n.VectorKeyframeTrack, y + ".position", u, "pos", o),
                    r(n.QuaternionKeyframeTrack, y + ".quaternion", u, "rot", o),
                    r(n.VectorKeyframeTrack, y + ".scale", u, "scl", o)
                }
        }
        if (0 === o.length)
            return null;
        var C = new n.AnimationClip(a,s,o);
        return C
    }
}),
n.AnimationMixer = function(e) {
    this._root = e,
    this._initMemoryManager(),
    this._accuIndex = 0,
    this.time = 0,
    this.timeScale = 1
}
,
n.AnimationMixer.prototype = {
    constructor: n.AnimationMixer,
    clipAction: function(e, t) {
        var i, r = t || this._root, o = r.uuid, a = "string" == typeof e ? e : e.name, s = e !== a ? e : null, l = this._actionsByClip[a];
        if (void 0 !== l) {
            var h = l.actionByRoot[o];
            if (void 0 !== h)
                return h;
            if (i = l.knownActions[0],
            s = i._clip,
            e !== a && e !== s)
                throw new Error("Different clips with the same name detected!")
        }
        if (null === s)
            return null;
        var c = new n.AnimationMixer._Action(this,s,t);
        return this._bindAction(c, i),
        this._addInactiveAction(c, a, o),
        c
    },
    existingAction: function(e, t) {
        var i = t || this._root
          , n = i.uuid
          , r = "string" == typeof e ? e : e.name
          , o = this._actionsByClip[r];
        return void 0 !== o ? o.actionByRoot[n] || null : null
    },
    stopAllAction: function() {
        var e = this._actions
          , t = this._nActiveActions
          , i = this._bindings
          , n = this._nActiveBindings;
        this._nActiveActions = 0,
        this._nActiveBindings = 0;
        for (var r = 0; r !== t; ++r)
            e[r].reset();
        for (var r = 0; r !== n; ++r)
            i[r].useCount = 0;
        return this
    },
    update: function(e) {
        e *= this.timeScale;
        for (var t = this._actions, i = this._nActiveActions, n = this.time += e, r = Math.sign(e), o = this._accuIndex ^= 1, a = 0; a !== i; ++a) {
            var s = t[a];
            s.enabled && s._update(n, e, r, o)
        }
        for (var l = this._bindings, h = this._nActiveBindings, a = 0; a !== h; ++a)
            l[a].apply(o);
        return this
    },
    getRoot: function() {
        return this._root
    },
    uncacheClip: function(e) {
        var t = this._actions
          , i = e.name
          , n = this._actionsByClip
          , r = n[i];
        if (void 0 !== r) {
            for (var o = r.knownActions, a = 0, s = o.length; a !== s; ++a) {
                var l = o[a];
                this._deactivateAction(l);
                var h = l._cacheIndex
                  , c = t[t.length - 1];
                l._cacheIndex = null,
                l._byClipCacheIndex = null,
                c._cacheIndex = h,
                t[h] = c,
                t.pop(),
                this._removeInactiveBindingsForAction(l)
            }
            delete n[i]
        }
    },
    uncacheRoot: function(e) {
        var t = e.uuid
          , i = this._actionsByClip;
        for (var n in i) {
            var r = i[n].actionByRoot
              , o = r[t];
            void 0 !== o && (this._deactivateAction(o),
            this._removeInactiveAction(o))
        }
        var a = this._bindingsByRootAndName
          , s = a[t];
        if (void 0 !== s)
            for (var l in s) {
                var h = s[l];
                h.restoreOriginalState(),
                this._removeInactiveBinding(h)
            }
    },
    uncacheAction: function(e, t) {
        var i = this.existingAction(e, t);
        null !== i && (this._deactivateAction(i),
        this._removeInactiveAction(i))
    }
},
n.EventDispatcher.prototype.apply(n.AnimationMixer.prototype),
n.AnimationMixer._Action = function(e, t, i) {
    this._mixer = e,
    this._clip = t,
    this._localRoot = i || null;
    for (var r = t.tracks, o = r.length, a = new Array(o), s = {
        endingStart: n.ZeroCurvatureEnding,
        endingEnd: n.ZeroCurvatureEnding
    }, l = 0; l !== o; ++l) {
        var h = r[l].createInterpolant(null);
        a[l] = h,
        h.settings = s
    }
    this._interpolantSettings = s,
    this._interpolants = a,
    this._propertyBindings = new Array(o),
    this._cacheIndex = null,
    this._byClipCacheIndex = null,
    this._timeScaleInterpolant = null,
    this._weightInterpolant = null,
    this.loop = n.LoopRepeat,
    this._loopCount = -1,
    this._startTime = null,
    this.time = 0,
    this.timeScale = 1,
    this._effectiveTimeScale = 1,
    this.weight = 1,
    this._effectiveWeight = 1,
    this.repetitions = 1 / 0,
    this.paused = !1,
    this.enabled = !0,
    this.clampWhenFinished = !1,
    this.zeroSlopeAtStart = !0,
    this.zeroSlopeAtEnd = !0
}
,
n.AnimationMixer._Action.prototype = {
    constructor: n.AnimationMixer._Action,
    play: function() {
        return this._mixer._activateAction(this),
        this
    },
    stop: function() {
        return this._mixer._deactivateAction(this),
        this.reset()
    },
    reset: function() {
        return this.paused = !1,
        this.enabled = !0,
        this.time = 0,
        this._loopCount = -1,
        this._startTime = null,
        this.stopFading().stopWarping()
    },
    isRunning: function() {
        this._startTime;
        return this.enabled && !this.paused && 0 !== this.timeScale && null === this._startTime && this._mixer._isActiveAction(this)
    },
    isScheduled: function() {
        return this._mixer._isActiveAction(this)
    },
    startAt: function(e) {
        return this._startTime = e,
        this
    },
    setLoop: function(e, t) {
        return this.loop = e,
        this.repetitions = t,
        this
    },
    setEffectiveWeight: function(e) {
        return this.weight = e,
        this._effectiveWeight = this.enabled ? e : 0,
        this.stopFading()
    },
    getEffectiveWeight: function() {
        return this._effectiveWeight
    },
    fadeIn: function(e) {
        return this._scheduleFading(e, 0, 1)
    },
    fadeOut: function(e) {
        return this._scheduleFading(e, 1, 0)
    },
    crossFadeFrom: function(e, t, i) {
        this._mixer;
        if (e.fadeOut(t),
        this.fadeIn(t),
        i) {
            var n = this._clip.duration
              , r = e._clip.duration
              , o = r / n
              , a = n / r;
            e.warp(1, o, t),
            this.warp(a, 1, t)
        }
        return this
    },
    crossFadeTo: function(e, t, i) {
        return e.crossFadeFrom(this, t, i)
    },
    stopFading: function() {
        var e = this._weightInterpolant;
        return null !== e && (this._weightInterpolant = null,
        this._mixer._takeBackControlInterpolant(e)),
        this
    },
    setEffectiveTimeScale: function(e) {
        return this.timeScale = e,
        this._effectiveTimeScale = this.paused ? 0 : e,
        this.stopWarping()
    },
    getEffectiveTimeScale: function() {
        return this._effectiveTimeScale
    },
    setDuration: function(e) {
        return this.timeScale = this._clip.duration / e,
        this.stopWarping()
    },
    syncWith: function(e) {
        return this.time = e.time,
        this.timeScale = e.timeScale,
        this.stopWarping()
    },
    halt: function(e) {
        return this.warp(this._currentTimeScale, 0, e)
    },
    warp: function(e, t, i) {
        var n = this._mixer
          , r = n.time
          , o = this._timeScaleInterpolant
          , a = this.timeScale;
        null === o && (o = n._lendControlInterpolant(),
        this._timeScaleInterpolant = o);
        var s = o.parameterPositions
          , l = o.sampleValues;
        return s[0] = r,
        s[1] = r + i,
        l[0] = e / a,
        l[1] = t / a,
        this
    },
    stopWarping: function() {
        var e = this._timeScaleInterpolant;
        return null !== e && (this._timeScaleInterpolant = null,
        this._mixer._takeBackControlInterpolant(e)),
        this
    },
    getMixer: function() {
        return this._mixer
    },
    getClip: function() {
        return this._clip
    },
    getRoot: function() {
        return this._localRoot || this._mixer._root
    },
    _update: function(e, t, i, n) {
        var r = this._startTime;
        if (null !== r) {
            var o = (e - r) * i;
            if (o < 0 || 0 === i)
                return;
            this._startTime = null,
            t = i * o
        }
        t *= this._updateTimeScale(e);
        var a = this._updateTime(t)
          , s = this._updateWeight(e);
        if (s > 0)
            for (var l = this._interpolants, h = this._propertyBindings, c = 0, u = l.length; c !== u; ++c)
                l[c].evaluate(a),
                h[c].accumulate(n, s)
    },
    _updateWeight: function(e) {
        var t = 0;
        if (this.enabled) {
            t = this.weight;
            var i = this._weightInterpolant;
            if (null !== i) {
                var n = i.evaluate(e)[0];
                t *= n,
                e > i.parameterPositions[1] && (this.stopFading(),
                0 === n && (this.enabled = !1))
            }
        }
        return this._effectiveWeight = t,
        t
    },
    _updateTimeScale: function(e) {
        var t = 0;
        if (!this.paused) {
            t = this.timeScale;
            var i = this._timeScaleInterpolant;
            if (null !== i) {
                var n = i.evaluate(e)[0];
                t *= n,
                e > i.parameterPositions[1] && (this.stopWarping(),
                0 === t ? this.pause = !0 : this.timeScale = t)
            }
        }
        return this._effectiveTimeScale = t,
        t
    },
    _updateTime: function(e) {
        var t = this.time + e;
        if (0 === e)
            return t;
        var i = this._clip.duration
          , r = this.loop
          , o = this._loopCount
          , a = !1;
        switch (r) {
        case n.LoopOnce:
            if (o === -1 && (this.loopCount = 0,
            this._setEndings(!0, !0, !1)),
            t >= i)
                t = i;
            else {
                if (!(t < 0))
                    break;
                t = 0
            }
            this.clampWhenFinished ? this.pause = !0 : this.enabled = !1,
            this._mixer.dispatchEvent({
                type: "finished",
                action: this,
                direction: e < 0 ? -1 : 1
            });
            break;
        case n.LoopPingPong:
            a = !0;
        case n.LoopRepeat:
            if (o === -1 && (e > 0 ? (o = 0,
            this._setEndings(!0, 0 === this.repetitions, a)) : this._setEndings(0 === this.repetitions, !0, a)),
            t >= i || t < 0) {
                var s = Math.floor(t / i);
                t -= i * s,
                o += Math.abs(s);
                var l = this.repetitions - o;
                if (l < 0) {
                    this.clampWhenFinished ? this.paused = !0 : this.enabled = !1,
                    t = e > 0 ? i : 0,
                    this._mixer.dispatchEvent({
                        type: "finished",
                        action: this,
                        direction: e > 0 ? 1 : -1
                    });
                    break
                }
                if (0 === l) {
                    var h = e < 0;
                    this._setEndings(h, !h, a)
                } else
                    this._setEndings(!1, !1, a);
                this._loopCount = o,
                this._mixer.dispatchEvent({
                    type: "loop",
                    action: this,
                    loopDelta: s
                })
            }
            if (r === n.LoopPingPong && 1 === (1 & o))
                return this.time = t,
                i - t
        }
        return this.time = t,
        t
    },
    _setEndings: function(e, t, i) {
        var r = this._interpolantSettings;
        i ? (r.endingStart = n.ZeroSlopeEnding,
        r.endingEnd = n.ZeroSlopeEnding) : (e ? r.endingStart = this.zeroSlopeAtStart ? n.ZeroSlopeEnding : n.ZeroCurvatureEnding : r.endingStart = n.WrapAroundEnding,
        t ? r.endingEnd = this.zeroSlopeAtEnd ? n.ZeroSlopeEnding : n.ZeroCurvatureEnding : r.endingEnd = n.WrapAroundEnding)
    },
    _scheduleFading: function(e, t, i) {
        var n = this._mixer
          , r = n.time
          , o = this._weightInterpolant;
        null === o && (o = n._lendControlInterpolant(),
        this._weightInterpolant = o);
        var a = o.parameterPositions
          , s = o.sampleValues;
        return a[0] = r,
        s[0] = t,
        a[1] = r + e,
        s[1] = i,
        this
    }
},
Object.assign(n.AnimationMixer.prototype, {
    _bindAction: function(e, t) {
        var i = e._localRoot || this._root
          , r = e._clip.tracks
          , o = r.length
          , a = e._propertyBindings
          , s = e._interpolants
          , l = i.uuid
          , h = this._bindingsByRootAndName
          , c = h[l];
        void 0 === c && (c = {},
        h[l] = c);
        for (var u = 0; u !== o; ++u) {
            var d = r[u]
              , p = d.name
              , f = c[p];
            if (void 0 !== f)
                a[u] = f;
            else {
                if (f = a[u],
                void 0 !== f) {
                    null === f._cacheIndex && (++f.referenceCount,
                    this._addInactiveBinding(f, l, p));
                    continue
                }
                var g = t && t._propertyBindings[u].binding.parsedPath;
                f = new n.PropertyMixer(n.PropertyBinding.create(i, p, g),d.ValueTypeName,d.getValueSize()),
                ++f.referenceCount,
                this._addInactiveBinding(f, l, p),
                a[u] = f
            }
            s[u].resultBuffer = f.buffer
        }
    },
    _activateAction: function(e) {
        if (!this._isActiveAction(e)) {
            if (null === e._cacheIndex) {
                var t = (e._localRoot || this._root).uuid
                  , i = e._clip.name
                  , n = this._actionsByClip[i];
                this._bindAction(e, n && n.knownActions[0]),
                this._addInactiveAction(e, i, t)
            }
            for (var r = e._propertyBindings, o = 0, a = r.length; o !== a; ++o) {
                var s = r[o];
                0 === s.useCount++ && (this._lendBinding(s),
                s.saveOriginalState())
            }
            this._lendAction(e)
        }
    },
    _deactivateAction: function(e) {
        if (this._isActiveAction(e)) {
            for (var t = e._propertyBindings, i = 0, n = t.length; i !== n; ++i) {
                var r = t[i];
                0 === --r.useCount && (r.restoreOriginalState(),
                this._takeBackBinding(r))
            }
            this._takeBackAction(e)
        }
    },
    _initMemoryManager: function() {
        this._actions = [],
        this._nActiveActions = 0,
        this._actionsByClip = {},
        this._bindings = [],
        this._nActiveBindings = 0,
        this._bindingsByRootAndName = {},
        this._controlInterpolants = [],
        this._nActiveControlInterpolants = 0;
        var e = this;
        this.stats = {
            actions: {
                get total() {
                    return e._actions.length
                },
                get inUse() {
                    return e._nActiveActions
                }
            },
            bindings: {
                get total() {
                    return e._bindings.length
                },
                get inUse() {
                    return e._nActiveBindings
                }
            },
            controlInterpolants: {
                get total() {
                    return e._controlInterpolants.length
                },
                get inUse() {
                    return e._nActiveControlInterpolants
                }
            }
        }
    },
    _isActiveAction: function(e) {
        var t = e._cacheIndex;
        return null !== t && t < this._nActiveActions
    },
    _addInactiveAction: function(e, t, i) {
        var n = this._actions
          , r = this._actionsByClip
          , o = r[t];
        if (void 0 === o)
            o = {
                knownActions: [e],
                actionByRoot: {}
            },
            e._byClipCacheIndex = 0,
            r[t] = o;
        else {
            var a = o.knownActions;
            e._byClipCacheIndex = a.length,
            a.push(e)
        }
        e._cacheIndex = n.length,
        n.push(e),
        o.actionByRoot[i] = e
    },
    _removeInactiveAction: function(e) {
        var t = this._actions
          , i = t[t.length - 1]
          , n = e._cacheIndex;
        i._cacheIndex = n,
        t[n] = i,
        t.pop(),
        e._cacheIndex = null;
        var r = e._clip.name
          , o = this._actionsByClip
          , a = o[r]
          , s = a.knownActions
          , l = s[s.length - 1]
          , h = e._byClipCacheIndex;
        l._byClipCacheIndex = h,
        s[h] = l,
        s.pop(),
        e._byClipCacheIndex = null;
        var c = a.actionByRoot
          , u = (t._localRoot || this._root).uuid;
        delete c[u],
        0 === s.length && delete o[r],
        this._removeInactiveBindingsForAction(e)
    },
    _removeInactiveBindingsForAction: function(e) {
        for (var t = e._propertyBindings, i = 0, n = t.length; i !== n; ++i) {
            var r = t[i];
            0 === --r.referenceCount && this._removeInactiveBinding(r)
        }
    },
    _lendAction: function(e) {
        var t = this._actions
          , i = e._cacheIndex
          , n = this._nActiveActions++
          , r = t[n];
        e._cacheIndex = n,
        t[n] = e,
        r._cacheIndex = i,
        t[i] = r
    },
    _takeBackAction: function(e) {
        var t = this._actions
          , i = e._cacheIndex
          , n = --this._nActiveActions
          , r = t[n];
        e._cacheIndex = n,
        t[n] = e,
        r._cacheIndex = i,
        t[i] = r
    },
    _addInactiveBinding: function(e, t, i) {
        var n = this._bindingsByRootAndName
          , r = n[t]
          , o = this._bindings;
        void 0 === r && (r = {},
        n[t] = r),
        r[i] = e,
        e._cacheIndex = o.length,
        o.push(e)
    },
    _removeInactiveBinding: function(e) {
        var t = this._bindings
          , i = e.binding
          , n = i.rootNode.uuid
          , r = i.path
          , o = this._bindingsByRootAndName
          , a = o[n]
          , s = t[t.length - 1]
          , l = e._cacheIndex;
        s._cacheIndex = l,
        t[l] = s,
        t.pop(),
        delete a[r];
        e: {
            for (var h in a)
                break e;
            delete o[n]
        }
    },
    _lendBinding: function(e) {
        var t = this._bindings
          , i = e._cacheIndex
          , n = this._nActiveBindings++
          , r = t[n];
        e._cacheIndex = n,
        t[n] = e,
        r._cacheIndex = i,
        t[i] = r
    },
    _takeBackBinding: function(e) {
        var t = this._bindings
          , i = e._cacheIndex
          , n = --this._nActiveBindings
          , r = t[n];
        e._cacheIndex = n,
        t[n] = e,
        r._cacheIndex = i,
        t[i] = r
    },
    _lendControlInterpolant: function() {
        var e = this._controlInterpolants
          , t = this._nActiveControlInterpolants++
          , i = e[t];
        return void 0 === i && (i = new n.LinearInterpolant(new Float32Array(2),new Float32Array(2),1,this._controlInterpolantsResultBuffer),
        i.__cacheIndex = t,
        e[t] = i),
        i
    },
    _takeBackControlInterpolant: function(e) {
        var t = this._controlInterpolants
          , i = e.__cacheIndex
          , n = --this._nActiveControlInterpolants
          , r = t[n];
        e.__cacheIndex = n,
        t[n] = e,
        r.__cacheIndex = i,
        t[i] = r
    },
    _controlInterpolantsResultBuffer: new Float32Array(1)
}),
n.AnimationObjectGroup = function(e) {
    this.uuid = n.Math.generateUUID(),
    this._objects = Array.prototype.slice.call(arguments),
    this.nCachedObjects_ = 0;
    var t = {};
    this._indicesByUUID = t;
    for (var i = 0, r = arguments.length; i !== r; ++i)
        t[arguments[i].uuid] = i;
    this._paths = [],
    this._parsedPaths = [],
    this._bindings = [],
    this._bindingsIndicesByPath = {};
    var o = this;
    this.stats = {
        objects: {
            get total() {
                return o._objects.length
            },
            get inUse() {
                return this.total - o.nCachedObjects_
            }
        },
        get bindingsPerObject() {
            return o._bindings.length
        }
    }
}
,
n.AnimationObjectGroup.prototype = {
    constructor: n.AnimationObjectGroup,
    add: function(e) {
        for (var t = this._objects, i = t.length, r = this.nCachedObjects_, o = this._indicesByUUID, a = this._paths, s = this._parsedPaths, l = this._bindings, h = l.length, c = 0, u = arguments.length; c !== u; ++c) {
            var d = arguments[c]
              , p = d.uuid
              , f = o[p];
            if (void 0 === f) {
                f = i++,
                o[p] = f,
                t.push(d);
                for (var g = 0, m = h; g !== m; ++g)
                    l[g].push(new n.PropertyBinding(d,a[g],s[g]))
            } else if (f < r) {
                var v = t[f]
                  , A = --r
                  , y = t[A];
                o[y.uuid] = f,
                t[f] = y,
                o[p] = A,
                t[A] = d;
                for (var g = 0, m = h; g !== m; ++g) {
                    var C = l[g]
                      , I = C[A]
                      , b = C[f];
                    C[f] = I,
                    void 0 === b && (b = new n.PropertyBinding(d,a[g],s[g])),
                    C[A] = b
                }
            } else
                t[f] !== v && console.error("Different objects with the same UUID detected. Clean the caches or recreate your infrastructure when reloading scenes...")
        }
        this.nCachedObjects_ = r
    },
    remove: function(e) {
        for (var t = this._objects, i = (t.length,
        this.nCachedObjects_), n = this._indicesByUUID, r = this._bindings, o = r.length, a = 0, s = arguments.length; a !== s; ++a) {
            var l = arguments[a]
              , h = l.uuid
              , c = n[h];
            if (void 0 !== c && c >= i) {
                var u = i++
                  , d = t[u];
                n[d.uuid] = c,
                t[c] = d,
                n[h] = u,
                t[u] = l;
                for (var p = 0, f = o; p !== f; ++p) {
                    var g = r[p]
                      , m = g[u]
                      , v = g[c];
                    g[c] = m,
                    g[u] = v
                }
            }
        }
        this.nCachedObjects_ = i
    },
    uncache: function(e) {
        for (var t = this._objects, i = t.length, n = this.nCachedObjects_, r = this._indicesByUUID, o = this._bindings, a = o.length, s = 0, l = arguments.length; s !== l; ++s) {
            var h = arguments[s]
              , c = h.uuid
              , u = r[c];
            if (void 0 !== u)
                if (delete r[c],
                u < n) {
                    var d = --n
                      , p = t[d]
                      , f = --i
                      , g = t[f];
                    r[p.uuid] = u,
                    t[u] = p,
                    r[g.uuid] = d,
                    t[d] = g,
                    t.pop();
                    for (var m = 0, v = a; m !== v; ++m) {
                        var A = o[m]
                          , y = A[d]
                          , C = A[f];
                        A[u] = y,
                        A[d] = C,
                        A.pop()
                    }
                } else {
                    var f = --i
                      , g = t[f];
                    r[g.uuid] = u,
                    t[u] = g,
                    t.pop();
                    for (var m = 0, v = a; m !== v; ++m) {
                        var A = o[m];
                        A[u] = A[f],
                        A.pop()
                    }
                }
        }
        this.nCachedObjects_ = n
    },
    subscribe_: function(e, t) {
        var i = this._bindingsIndicesByPath
          , r = i[e]
          , o = this._bindings;
        if (void 0 !== r)
            return o[r];
        var a = this._paths
          , s = this._parsedPaths
          , l = this._objects
          , h = l.length
          , c = this.nCachedObjects_
          , u = new Array(h);
        r = o.length,
        i[e] = r,
        a.push(e),
        s.push(t),
        o.push(u);
        for (var d = c, p = l.length; d !== p; ++d) {
            var f = l[d];
            u[d] = new n.PropertyBinding(f,e,t)
        }
        return u
    },
    unsubscribe_: function(e) {
        var t = this._bindingsIndicesByPath
          , i = t[e];
        if (void 0 !== i) {
            var n = this._paths
              , r = this._parsedPaths
              , o = this._bindings
              , a = o.length - 1
              , s = o[a]
              , l = e[a];
            t[l] = i,
            o[i] = s,
            o.pop(),
            r[i] = r[a],
            r.pop(),
            n[i] = n[a],
            n.pop()
        }
    }
},
n.AnimationUtils = {
    arraySlice: function(e, t, i) {
        return n.AnimationUtils.isTypedArray(e) ? new e.constructor(e.subarray(t, i)) : e.slice(t, i)
    },
    convertArray: function(e, t, i) {
        return !e || !i && e.constructor === t ? e : "number" == typeof t.BYTES_PER_ELEMENT ? new t(e) : Array.prototype.slice.call(e)
    },
    isTypedArray: function(e) {
        return ArrayBuffer.isView(e) && !(e instanceof DataView)
    },
    getKeyframeOrder: function(e) {
        function t(t, i) {
            return e[t] - e[i]
        }
        for (var i = e.length, n = new Array(i), r = 0; r !== i; ++r)
            n[r] = r;
        return n.sort(t),
        n
    },
    sortedArray: function(e, t, i) {
        for (var n = e.length, r = new e.constructor(n), o = 0, a = 0; a !== n; ++o)
            for (var s = i[o] * t, l = 0; l !== t; ++l)
                r[a++] = e[s + l];
        return r
    },
    flattenJSON: function(e, t, i, n) {
        for (var r = 1, o = e[0]; void 0 !== o && void 0 === o[n]; )
            o = e[r++];
        if (void 0 !== o) {
            var a = o[n];
            if (void 0 !== a)
                if (Array.isArray(a)) {
                    do
                        a = o[n],
                        void 0 !== a && (t.push(o.time),
                        i.push.apply(i, a)),
                        o = e[r++];
                    while (void 0 !== o)
                } else if (void 0 !== a.toArray) {
                    do
                        a = o[n],
                        void 0 !== a && (t.push(o.time),
                        a.toArray(i, i.length)),
                        o = e[r++];
                    while (void 0 !== o)
                } else
                    do
                        a = o[n],
                        void 0 !== a && (t.push(o.time),
                        i.push(a)),
                        o = e[r++];
                    while (void 0 !== o)
        }
    }
},
n.KeyframeTrack = function(e, t, i, r) {
    if (void 0 === e)
        throw new Error("track name is undefined");
    if (void 0 === t || 0 === t.length)
        throw new Error("no keyframes in track named " + e);
    this.name = e,
    this.times = n.AnimationUtils.convertArray(t, this.TimeBufferType),
    this.values = n.AnimationUtils.convertArray(i, this.ValueBufferType),
    this.setInterpolation(r || this.DefaultInterpolation),
    this.validate(),
    this.optimize()
}
,
n.KeyframeTrack.prototype = {
    constructor: n.KeyframeTrack,
    TimeBufferType: Float32Array,
    ValueBufferType: Float32Array,
    DefaultInterpolation: n.InterpolateLinear,
    InterpolantFactoryMethodDiscrete: function(e) {
        return new n.DiscreteInterpolant(this.times,this.values,this.getValueSize(),e)
    },
    InterpolantFactoryMethodLinear: function(e) {
        return new n.LinearInterpolant(this.times,this.values,this.getValueSize(),e)
    },
    InterpolantFactoryMethodSmooth: function(e) {
        return new n.CubicInterpolant(this.times,this.values,this.getValueSize(),e)
    },
    setInterpolation: function(e) {
        var t = void 0;
        switch (e) {
        case n.InterpolateDiscrete:
            t = this.InterpolantFactoryMethodDiscrete;
            break;
        case n.InterpolateLinear:
            t = this.InterpolantFactoryMethodLinear;
            break;
        case n.InterpolateSmooth:
            t = this.InterpolantFactoryMethodSmooth
        }
        if (void 0 === t) {
            var i = "unsupported interpolation for " + this.ValueTypeName + " keyframe track named " + this.name;
            if (void 0 === this.createInterpolant) {
                if (e === this.DefaultInterpolation)
                    throw new Error(i);
                this.setInterpolation(this.DefaultInterpolation)
            }
            return void console.warn(i)
        }
        this.createInterpolant = t
    },
    getInterpolation: function() {
        switch (this.createInterpolant) {
        case this.InterpolantFactoryMethodDiscrete:
            return n.InterpolateDiscrete;
        case this.InterpolantFactoryMethodLinear:
            return n.InterpolateLinear;
        case this.InterpolantFactoryMethodSmooth:
            return n.InterpolateSmooth
        }
    },
    getValueSize: function() {
        return this.values.length / this.times.length
    },
    shift: function(e) {
        if (0 !== e)
            for (var t = this.times, i = 0, n = t.length; i !== n; ++i)
                t[i] += e;
        return this
    },
    scale: function(e) {
        if (1 !== e)
            for (var t = this.times, i = 0, n = t.length; i !== n; ++i)
                t[i] *= e;
        return this
    },
    trim: function(e, t) {
        for (var i = this.times, r = i.length, o = 0, a = r - 1; o !== r && i[o] < e; )
            ++o;
        for (; a !== -1 && i[a] > t; )
            --a;
        if (++a,
        0 !== o || a !== r) {
            o >= a && (a = Math.max(a, 1),
            o = a - 1);
            var s = this.getValueSize();
            this.times = n.AnimationUtils.arraySlice(i, o, a),
            this.values = n.AnimationUtils.arraySlice(this.values, o * s, a * s)
        }
        return this
    },
    validate: function() {
        var e = !0
          , t = this.getValueSize();
        t - Math.floor(t) !== 0 && (console.error("invalid value size in track", this),
        e = !1);
        var i = this.times
          , r = this.values
          , o = i.length;
        0 === o && (console.error("track is empty", this),
        e = !1);
        for (var a = null, s = 0; s !== o; s++) {
            var l = i[s];
            if ("number" == typeof l && isNaN(l)) {
                console.error("time is not a valid number", this, s, l),
                e = !1;
                break
            }
            if (null !== a && a > l) {
                console.error("out of order keys", this, s, l, a),
                e = !1;
                break
            }
            a = l
        }
        if (void 0 !== r && n.AnimationUtils.isTypedArray(r))
            for (var s = 0, h = r.length; s !== h; ++s) {
                var c = r[s];
                if (isNaN(c)) {
                    console.error("value is not a valid number", this, s, c),
                    e = !1;
                    break
                }
            }
        return e
    },
    optimize: function() {
        for (var e = this.times, t = this.values, i = this.getValueSize(), r = 1, o = 1, a = e.length - 1; o <= a; ++o) {
            var s = !1
              , l = e[o]
              , h = e[o + 1];
            if (l !== h && (1 !== o || l !== l[0]))
                for (var c = o * i, u = c - i, d = c + i, p = 0; p !== i; ++p) {
                    var f = t[c + p];
                    if (f !== t[u + p] || f !== t[d + p]) {
                        s = !0;
                        break
                    }
                }
            if (s) {
                if (o !== r) {
                    e[r] = e[o];
                    for (var g = o * i, m = r * i, p = 0; p !== i; ++p)
                        t[m + p] = t[g + p]
                }
                ++r
            }
        }
        return r !== e.length && (this.times = n.AnimationUtils.arraySlice(e, 0, r),
        this.values = n.AnimationUtils.arraySlice(t, 0, r * i)),
        this
    }
},
Object.assign(n.KeyframeTrack, {
    parse: function(e) {
        if (void 0 === e.type)
            throw new Error("track type undefined, can not parse");
        var t = n.KeyframeTrack._getTrackTypeForValueTypeName(e.type);
        if (void 0 === e.times) {
            console.warn("legacy JSON format detected, converting");
            var i = []
              , r = [];
            n.AnimationUtils.flattenJSON(e.keys, i, r, "value"),
            e.times = i,
            e.values = r
        }
        return void 0 !== t.parse ? t.parse(e) : new t(e.name,e.times,e.values,e.interpolation)
    },
    toJSON: function(e) {
        var t, i = e.constructor;
        if (void 0 !== i.toJSON)
            t = i.toJSON(e);
        else {
            t = {
                name: e.name,
                times: n.AnimationUtils.convertArray(e.times, Array),
                values: n.AnimationUtils.convertArray(e.values, Array)
            };
            var r = e.getInterpolation();
            r !== e.DefaultInterpolation && (t.interpolation = r)
        }
        return t.type = e.ValueTypeName,
        t
    },
    _getTrackTypeForValueTypeName: function(e) {
        switch (e.toLowerCase()) {
        case "scalar":
        case "double":
        case "float":
        case "number":
        case "integer":
            return n.NumberKeyframeTrack;
        case "vector":
        case "vector2":
        case "vector3":
        case "vector4":
            return n.VectorKeyframeTrack;
        case "color":
            return n.ColorKeyframeTrack;
        case "quaternion":
            return n.QuaternionKeyframeTrack;
        case "bool":
        case "boolean":
            return n.BooleanKeyframeTrack;
        case "string":
            return n.StringKeyframeTrack
        }
        throw new Error("Unsupported typeName: " + e)
    }
}),
n.PropertyBinding = function(e, t, i) {
    this.path = t,
    this.parsedPath = i || n.PropertyBinding.parseTrackName(t),
    this.node = n.PropertyBinding.findNode(e, this.parsedPath.nodeName) || e,
    this.rootNode = e
}
,
n.PropertyBinding.prototype = {
    constructor: n.PropertyBinding,
    getValue: function(e, t) {
        this.bind(),
        this.getValue(e, t)
    },
    setValue: function(e, t) {
        this.bind(),
        this.setValue(e, t)
    },
    bind: function() {
        var e = this.node
          , t = this.parsedPath
          , i = t.objectName
          , r = t.propertyName
          , o = t.propertyIndex;
        if (e || (e = n.PropertyBinding.findNode(this.rootNode, t.nodeName) || this.rootNode,
        this.node = e),
        this.getValue = this._getValue_unavailable,
        this.setValue = this._setValue_unavailable,
        !e)
            return void console.error("  trying to update node for track: " + this.path + " but it wasn't found.");
        if (i) {
            var a = t.objectIndex;
            switch (i) {
            case "materials":
                if (!e.material)
                    return void console.error("  can not bind to material as node does not have a material", this);
                if (!e.material.materials)
                    return void console.error("  can not bind to material.materials as node.material does not have a materials array", this);
                e = e.material.materials;
                break;
            case "bones":
                if (!e.skeleton)
                    return void console.error("  can not bind to bones as node does not have a skeleton", this);
                e = e.skeleton.bones;
                for (var s = 0; s < e.length; s++)
                    if (e[s].name === a) {
                        a = s;
                        break
                    }
                break;
            default:
                if (void 0 === e[i])
                    return void console.error("  can not bind to objectName of node, undefined", this);
                e = e[i]
            }
            if (void 0 !== a) {
                if (void 0 === e[a])
                    return void console.error("  trying to bind to objectIndex of objectName, but is undefined:", this, e);
                e = e[a]
            }
        }
        var l = e[r];
        if (!l) {
            var h = t.nodeName;
            return void console.error("  trying to update property for track: " + h + "." + r + " but it wasn't found.", e)
        }
        var c = this.Versioning.None;
        void 0 !== e.needsUpdate ? (c = this.Versioning.NeedsUpdate,
        this.targetObject = e) : void 0 !== e.matrixWorldNeedsUpdate && (c = this.Versioning.MatrixWorldNeedsUpdate,
        this.targetObject = e);
        var u = this.BindingType.Direct;
        if (void 0 !== o) {
            if ("morphTargetInfluences" === r) {
                if (!e.geometry)
                    return void console.error("  can not bind to morphTargetInfluences becasuse node does not have a geometry", this);
                if (!e.geometry.morphTargets)
                    return void console.error("  can not bind to morphTargetInfluences becasuse node does not have a geometry.morphTargets", this);
                for (var s = 0; s < this.node.geometry.morphTargets.length; s++)
                    if (e.geometry.morphTargets[s].name === o) {
                        o = s;
                        break
                    }
            }
            u = this.BindingType.ArrayElement,
            this.resolvedProperty = l,
            this.propertyIndex = o
        } else
            void 0 !== l.fromArray && void 0 !== l.toArray ? (u = this.BindingType.HasFromToArray,
            this.resolvedProperty = l) : void 0 !== l.length ? (u = this.BindingType.EntireArray,
            this.resolvedProperty = l) : this.propertyName = r;
        this.getValue = this.GetterByBindingType[u],
        this.setValue = this.SetterByBindingTypeAndVersioning[u][c]
    },
    unbind: function() {
        this.node = null,
        this.getValue = this._getValue_unbound,
        this.setValue = this._setValue_unbound
    }
},
Object.assign(n.PropertyBinding.prototype, {
    _getValue_unavailable: function() {},
    _setValue_unavailable: function() {},
    _getValue_unbound: n.PropertyBinding.prototype.getValue,
    _setValue_unbound: n.PropertyBinding.prototype.setValue,
    BindingType: {
        Direct: 0,
        EntireArray: 1,
        ArrayElement: 2,
        HasFromToArray: 3
    },
    Versioning: {
        None: 0,
        NeedsUpdate: 1,
        MatrixWorldNeedsUpdate: 2
    },
    GetterByBindingType: [function(e, t) {
        e[t] = this.node[this.propertyName]
    }
    , function(e, t) {
        for (var i = this.resolvedProperty, n = 0, r = i.length; n !== r; ++n)
            e[t++] = i[n]
    }
    , function(e, t) {
        e[t] = this.resolvedProperty[this.propertyIndex]
    }
    , function(e, t) {
        this.resolvedProperty.toArray(e, t)
    }
    ],
    SetterByBindingTypeAndVersioning: [[function(e, t) {
        this.node[this.propertyName] = e[t]
    }
    , function(e, t) {
        this.node[this.propertyName] = e[t],
        this.targetObject.needsUpdate = !0
    }
    , function(e, t) {
        this.node[this.propertyName] = e[t],
        this.targetObject.matrixWorldNeedsUpdate = !0
    }
    ], [function(e, t) {
        for (var i = this.resolvedProperty, n = 0, r = i.length; n !== r; ++n)
            i[n] = e[t++]
    }
    , function(e, t) {
        for (var i = this.resolvedProperty, n = 0, r = i.length; n !== r; ++n)
            i[n] = e[t++];
        this.targetObject.needsUpdate = !0
    }
    , function(e, t) {
        for (var i = this.resolvedProperty, n = 0, r = i.length; n !== r; ++n)
            i[n] = e[t++];
        this.targetObject.matrixWorldNeedsUpdate = !0
    }
    ], [function(e, t) {
        this.resolvedProperty[this.propertyIndex] = e[t]
    }
    , function(e, t) {
        this.resolvedProperty[this.propertyIndex] = e[t],
        this.targetObject.needsUpdate = !0
    }
    , function(e, t) {
        this.resolvedProperty[this.propertyIndex] = e[t],
        this.targetObject.matrixWorldNeedsUpdate = !0
    }
    ], [function(e, t) {
        this.resolvedProperty.fromArray(e, t)
    }
    , function(e, t) {
        this.resolvedProperty.fromArray(e, t),
        this.targetObject.needsUpdate = !0
    }
    , function(e, t) {
        this.resolvedProperty.fromArray(e, t),
        this.targetObject.matrixWorldNeedsUpdate = !0
    }
    ]]
}),
n.PropertyBinding.Composite = function(e, t, i) {
    var r = i || n.PropertyBinding.parseTrackName(t);
    this._targetGroup = e,
    this._bindings = e.subscribe_(t, r)
}
,
n.PropertyBinding.Composite.prototype = {
    constructor: n.PropertyBinding.Composite,
    getValue: function(e, t) {
        this.bind();
        var i = this._targetGroup.nCachedObjects_
          , n = this._bindings[i];
        void 0 !== n && n.getValue(e, t)
    },
    setValue: function(e, t) {
        for (var i = this._bindings, n = this._targetGroup.nCachedObjects_, r = i.length; n !== r; ++n)
            i[n].setValue(e, t)
    },
    bind: function() {
        for (var e = this._bindings, t = this._targetGroup.nCachedObjects_, i = e.length; t !== i; ++t)
            e[t].bind()
    },
    unbind: function() {
        for (var e = this._bindings, t = this._targetGroup.nCachedObjects_, i = e.length; t !== i; ++t)
            e[t].unbind()
    }
},
n.PropertyBinding.create = function(e, t, i) {
    return e instanceof n.AnimationObjectGroup ? new n.PropertyBinding.Composite(e,t,i) : new n.PropertyBinding(e,t,i)
}
,
n.PropertyBinding.parseTrackName = function(e) {
    var t = /^(([\w]+\/)*)([\w-\d]+)?(\.([\w]+)(\[([\w\d\[\]\_.:\- ]+)\])?)?(\.([\w.]+)(\[([\w\d\[\]\_. ]+)\])?)$/
      , i = t.exec(e);
    if (!i)
        throw new Error("cannot parse trackName at all: " + e);
    i.index === t.lastIndex && t.lastIndex++;
    var n = {
        nodeName: i[3],
        objectName: i[5],
        objectIndex: i[7],
        propertyName: i[9],
        propertyIndex: i[11]
    };
    if (null === n.propertyName || 0 === n.propertyName.length)
        throw new Error("can not parse propertyName from trackName: " + e);
    return n
}
,
n.PropertyBinding.findNode = function(e, t) {
    if (!t || "" === t || "root" === t || "." === t || t === -1 || t === e.name || t === e.uuid)
        return e;
    if (e.skeleton) {
        var i = function(e) {
            for (var i = 0; i < e.bones.length; i++) {
                var n = e.bones[i];
                if (n.name === t)
                    return n
            }
            return null
        }
          , n = i(e.skeleton);
        if (n)
            return n
    }
    if (e.children) {
        var r = function(e) {
            for (var i = 0; i < e.length; i++) {
                var n = e[i];
                if (n.name === t || n.uuid === t)
                    return n;
                var o = r(n.children);
                if (o)
                    return o
            }
            return null
        }
          , o = r(e.children);
        if (o)
            return o
    }
    return null
}
,
n.PropertyMixer = function(e, t, i) {
    this.binding = e,
    this.valueSize = i;
    var n, r = Float64Array;
    switch (t) {
    case "quaternion":
        n = this._slerp;
        break;
    case "string":
    case "bool":
        r = Array,
        n = this._select;
        break;
    default:
        n = this._lerp
    }
    this.buffer = new r(4 * i),
    this._mixBufferRegion = n,
    this.cumulativeWeight = 0,
    this.useCount = 0,
    this.referenceCount = 0
}
,
n.PropertyMixer.prototype = {
    constructor: n.PropertyMixer,
    accumulate: function(e, t) {
        var i = this.buffer
          , n = this.valueSize
          , r = e * n + n
          , o = this.cumulativeWeight;
        if (0 === o) {
            for (var a = 0; a !== n; ++a)
                i[r + a] = i[a];
            o = t
        } else {
            o += t;
            var s = t / o;
            this._mixBufferRegion(i, r, 0, s, n)
        }
        this.cumulativeWeight = o
    },
    apply: function(e) {
        var t = this.valueSize
          , i = this.buffer
          , n = e * t + t
          , r = this.cumulativeWeight
          , o = this.binding;
        if (this.cumulativeWeight = 0,
        r < 1) {
            var a = 3 * t;
            this._mixBufferRegion(i, n, a, 1 - r, t)
        }
        for (var s = t, l = t + t; s !== l; ++s)
            if (i[s] !== i[s + t]) {
                o.setValue(i, n);
                break
            }
    },
    saveOriginalState: function() {
        var e = this.binding
          , t = this.buffer
          , i = this.valueSize
          , n = 3 * i;
        e.getValue(t, n);
        for (var r = i, o = n; r !== o; ++r)
            t[r] = t[n + r % i];
        this.cumulativeWeight = 0
    },
    restoreOriginalState: function() {
        var e = 3 * this.valueSize;
        this.binding.setValue(this.buffer, e)
    },
    _select: function(e, t, i, n, r) {
        if (n >= .5)
            for (var o = 0; o !== r; ++o)
                e[t + o] = e[i + o]
    },
    _slerp: function(e, t, i, r, o) {
        n.Quaternion.slerpFlat(e, t, e, t, e, i, r)
    },
    _lerp: function(e, t, i, n, r) {
        for (var o = 1 - n, a = 0; a !== r; ++a) {
            var s = t + a;
            e[s] = e[s] * o + e[i + a] * n
        }
    }
},
n.BooleanKeyframeTrack = function(e, t, i) {
    n.KeyframeTrack.call(this, e, t, i)
}
,
n.BooleanKeyframeTrack.prototype = Object.assign(Object.create(n.KeyframeTrack.prototype), {
    constructor: n.BooleanKeyframeTrack,
    ValueTypeName: "bool",
    ValueBufferType: Array,
    DefaultInterpolation: n.IntepolateDiscrete,
    InterpolantFactoryMethodLinear: void 0,
    InterpolantFactoryMethodSmooth: void 0
}),
n.NumberKeyframeTrack = function(e, t, i, r) {
    n.KeyframeTrack.call(this, e, t, i, r)
}
,
n.NumberKeyframeTrack.prototype = Object.assign(Object.create(n.KeyframeTrack.prototype), {
    constructor: n.NumberKeyframeTrack,
    ValueTypeName: "number"
}),
n.QuaternionKeyframeTrack = function(e, t, i, r) {
    n.KeyframeTrack.call(this, e, t, i, r)
}
,
n.QuaternionKeyframeTrack.prototype = Object.assign(Object.create(n.KeyframeTrack.prototype), {
    constructor: n.QuaternionKeyframeTrack,
    ValueTypeName: "quaternion",
    DefaultInterpolation: n.InterpolateLinear,
    InterpolantFactoryMethodLinear: function(e) {
        return new n.QuaternionLinearInterpolant(this.times,this.values,this.getValueSize(),e)
    },
    InterpolantFactoryMethodSmooth: void 0
}),
n.StringKeyframeTrack = function(e, t, i, r) {
    n.KeyframeTrack.call(this, e, t, i, r)
}
,
n.StringKeyframeTrack.prototype = Object.assign(Object.create(n.KeyframeTrack.prototype), {
    constructor: n.StringKeyframeTrack,
    ValueTypeName: "string",
    ValueBufferType: Array,
    DefaultInterpolation: n.IntepolateDiscrete,
    InterpolantFactoryMethodLinear: void 0,
    InterpolantFactoryMethodSmooth: void 0
}),
n.VectorKeyframeTrack = function(e, t, i, r) {
    n.KeyframeTrack.call(this, e, t, i, r)
}
,
n.VectorKeyframeTrack.prototype = Object.assign(Object.create(n.KeyframeTrack.prototype), {
    constructor: n.VectorKeyframeTrack,
    ValueTypeName: "vector"
}),
n.Audio = function(e) {
    n.Object3D.call(this),
    this.type = "Audio",
    this.context = e.context,
    this.source = this.context.createBufferSource(),
    this.source.onended = this.onEnded.bind(this),
    this.gain = this.context.createGain(),
    this.gain.connect(e.getInput()),
    this.autoplay = !1,
    this.startTime = 0,
    this.playbackRate = 1,
    this.isPlaying = !1,
    this.hasPlaybackControl = !0,
    this.sourceType = "empty",
    this.filter = null
}
,
n.Audio.prototype = Object.create(n.Object3D.prototype),
n.Audio.prototype.constructor = n.Audio,
n.Audio.prototype.getOutput = function() {
    return this.gain
}
,
n.Audio.prototype.load = function(e) {
    var t = new n.AudioBuffer(this.context);
    return t.load(e),
    this.setBuffer(t),
    this
}
,
n.Audio.prototype.setNodeSource = function(e) {
    return this.hasPlaybackControl = !1,
    this.sourceType = "audioNode",
    this.source = e,
    this.connect(),
    this
}
,
n.Audio.prototype.setBuffer = function(e) {
    var t = this;
    return e.onReady(function(e) {
        t.source.buffer = e,
        t.sourceType = "buffer",
        t.autoplay && t.play()
    }),
    this
}
,
n.Audio.prototype.play = function() {
    if (this.isPlaying === !0)
        return void console.warn("THREE.Audio: Audio is already playing.");
    if (this.hasPlaybackControl === !1)
        return void console.warn("THREE.Audio: this Audio has no playback control.");
    var e = this.context.createBufferSource();
    e.buffer = this.source.buffer,
    e.loop = this.source.loop,
    e.onended = this.source.onended,
    e.start(0, this.startTime),
    e.playbackRate.value = this.playbackRate,
    this.isPlaying = !0,
    this.source = e,
    this.connect()
}
,
n.Audio.prototype.pause = function() {
    return this.hasPlaybackControl === !1 ? void console.warn("THREE.Audio: this Audio has no playback control.") : (this.source.stop(),
    void (this.startTime = this.context.currentTime))
}
,
n.Audio.prototype.stop = function() {
    return this.hasPlaybackControl === !1 ? void console.warn("THREE.Audio: this Audio has no playback control.") : (this.source.stop(),
    void (this.startTime = 0))
}
,
n.Audio.prototype.connect = function() {
    null !== this.filter ? (this.source.connect(this.filter),
    this.filter.connect(this.getOutput())) : this.source.connect(this.getOutput())
}
,
n.Audio.prototype.disconnect = function() {
    null !== this.filter ? (this.source.disconnect(this.filter),
    this.filter.disconnect(this.getOutput())) : this.source.disconnect(this.getOutput())
}
,
n.Audio.prototype.getFilter = function() {
    return this.filter
}
,
n.Audio.prototype.setFilter = function(e) {
    void 0 === e && (e = null),
    this.isPlaying === !0 ? (this.disconnect(),
    this.filter = e,
    this.connect()) : this.filter = e
}
,
n.Audio.prototype.setPlaybackRate = function(e) {
    return this.hasPlaybackControl === !1 ? void console.warn("THREE.Audio: this Audio has no playback control.") : (this.playbackRate = e,
    void (this.isPlaying === !0 && (this.source.playbackRate.value = this.playbackRate)))
}
,
n.Audio.prototype.getPlaybackRate = function() {
    return this.playbackRate
}
,
n.Audio.prototype.onEnded = function() {
    this.isPlaying = !1
}
,
n.Audio.prototype.setLoop = function(e) {
    return this.hasPlaybackControl === !1 ? void console.warn("THREE.Audio: this Audio has no playback control.") : void (this.source.loop = e)
}
,
n.Audio.prototype.getLoop = function() {
    return this.hasPlaybackControl === !1 ? (console.warn("THREE.Audio: this Audio has no playback control."),
    !1) : this.source.loop
}
,
n.Audio.prototype.setVolume = function(e) {
    this.gain.gain.value = e
}
,
n.Audio.prototype.getVolume = function() {
    return this.gain.gain.value
}
,
n.AudioAnalyser = function(e, t) {
    this.analyser = e.context.createAnalyser(),
    this.analyser.fftSize = void 0 !== t ? t : 2048,
    this.data = new Uint8Array(this.analyser.frequencyBinCount),
    e.getOutput().connect(this.analyser)
}
,
n.AudioAnalyser.prototype = {
    constructor: n.AudioAnalyser,
    getData: function() {
        return this.analyser.getByteFrequencyData(this.data),
        this.data
    }
},
n.AudioBuffer = function(e) {
    this.context = e,
    this.ready = !1,
    this.readyCallbacks = []
}
,
n.AudioBuffer.prototype.load = function(e) {
    var t = this
      , i = new XMLHttpRequest;
    return i.open("GET", e, !0),
    i.responseType = "arraybuffer",
    i.onload = function(e) {
        t.context.decodeAudioData(this.response, function(e) {
            t.buffer = e,
            t.ready = !0;
            for (var i = 0; i < t.readyCallbacks.length; i++)
                t.readyCallbacks[i](t.buffer);
            t.readyCallbacks = []
        })
    }
    ,
    i.send(),
    this
}
,
n.AudioBuffer.prototype.onReady = function(e) {
    this.ready ? e(this.buffer) : this.readyCallbacks.push(e)
}
,
n.PositionalAudio = function(e) {
    n.Audio.call(this, e),
    this.panner = this.context.createPanner(),
    this.panner.connect(this.gain)
}
,
n.PositionalAudio.prototype = Object.create(n.Audio.prototype),
n.PositionalAudio.prototype.constructor = n.PositionalAudio,
n.PositionalAudio.prototype.getOutput = function() {
    return this.panner
}
,
n.PositionalAudio.prototype.setRefDistance = function(e) {
    this.panner.refDistance = e
}
,
n.PositionalAudio.prototype.getRefDistance = function() {
    return this.panner.refDistance
}
,
n.PositionalAudio.prototype.setRolloffFactor = function(e) {
    this.panner.rolloffFactor = e
}
,
n.PositionalAudio.prototype.getRolloffFactor = function() {
    return this.panner.rolloffFactor
}
,
n.PositionalAudio.prototype.setDistanceModel = function(e) {
    this.panner.distanceModel = e
}
,
n.PositionalAudio.prototype.getDistanceModel = function() {
    return this.panner.distanceModel
}
,
n.PositionalAudio.prototype.setMaxDistance = function(e) {
    this.panner.maxDistance = e
}
,
n.PositionalAudio.prototype.getMaxDistance = function() {
    return this.panner.maxDistance
}
,
n.PositionalAudio.prototype.updateMatrixWorld = function() {
    var e = new n.Vector3;
    return function(t) {
        n.Object3D.prototype.updateMatrixWorld.call(this, t),
        e.setFromMatrixPosition(this.matrixWorld),
        this.panner.setPosition(e.x, e.y, e.z)
    }
}(),
n.AudioListener = function() {
    n.Object3D.call(this),
    this.type = "AudioListener",
    this.context = new (window.AudioContext || window.webkitAudioContext),
    this.gain = this.context.createGain(),
    this.gain.connect(this.context.destination),
    this.filter = null
}
,
n.AudioListener.prototype = Object.create(n.Object3D.prototype),
n.AudioListener.prototype.constructor = n.AudioListener,
n.AudioListener.prototype.getInput = function() {
    return this.gain
}
,
n.AudioListener.prototype.removeFilter = function() {
    null !== this.filter && (this.gain.disconnect(this.filter),
    this.filter.disconnect(this.context.destination),
    this.gain.connect(this.context.destination),
    this.filter = null)
}
,
n.AudioListener.prototype.setFilter = function(e) {
    null !== this.filter ? (this.gain.disconnect(this.filter),
    this.filter.disconnect(this.context.destination)) : this.gain.disconnect(this.context.destination),
    this.filter = e,
    this.gain.connect(this.filter),
    this.filter.connect(this.context.destination)
}
,
n.AudioListener.prototype.getFilter = function() {
    return this.filter
}
,
n.AudioListener.prototype.setMasterVolume = function(e) {
    this.gain.gain.value = e
}
,
n.AudioListener.prototype.getMasterVolume = function() {
    return this.gain.gain.value
}
,
n.AudioListener.prototype.updateMatrixWorld = function() {
    var e = new n.Vector3
      , t = new n.Quaternion
      , i = new n.Vector3
      , r = new n.Vector3;
    return function(o) {
        n.Object3D.prototype.updateMatrixWorld.call(this, o);
        var a = this.context.listener
          , s = this.up;
        this.matrixWorld.decompose(e, t, i),
        r.set(0, 0, -1).applyQuaternion(t),
        a.setPosition(e.x, e.y, e.z),
        a.setOrientation(r.x, r.y, r.z, s.x, s.y, s.z)
    }
}(),
n.Camera = function() {
    n.Object3D.call(this),
    this.type = "Camera",
    this.matrixWorldInverse = new n.Matrix4,
    this.projectionMatrix = new n.Matrix4
}
,
n.Camera.prototype = Object.create(n.Object3D.prototype),
n.Camera.prototype.constructor = n.Camera,
n.Camera.prototype.getWorldDirection = function() {
    var e = new n.Quaternion;
    return function(t) {
        var i = t || new n.Vector3;
        return this.getWorldQuaternion(e),
        i.set(0, 0, -1).applyQuaternion(e)
    }
}(),
n.Camera.prototype.lookAt = function() {
    var e = new n.Matrix4;
    return function(t) {
        e.lookAt(this.position, t, this.up),
        this.quaternion.setFromRotationMatrix(e)
    }
}(),
n.Camera.prototype.clone = function() {
    return (new this.constructor).copy(this)
}
,
n.Camera.prototype.copy = function(e) {
    return n.Object3D.prototype.copy.call(this, e),
    this.matrixWorldInverse.copy(e.matrixWorldInverse),
    this.projectionMatrix.copy(e.projectionMatrix),
    this
}
,
n.CubeCamera = function(e, t, i) {
    n.Object3D.call(this),
    this.type = "CubeCamera";
    var r = 90
      , o = 1
      , a = new n.PerspectiveCamera(r,o,e,t);
    a.up.set(0, -1, 0),
    a.lookAt(new n.Vector3(1,0,0)),
    this.add(a);
    var s = new n.PerspectiveCamera(r,o,e,t);
    s.up.set(0, -1, 0),
    s.lookAt(new n.Vector3(-1,0,0)),
    this.add(s);
    var l = new n.PerspectiveCamera(r,o,e,t);
    l.up.set(0, 0, 1),
    l.lookAt(new n.Vector3(0,1,0)),
    this.add(l);
    var h = new n.PerspectiveCamera(r,o,e,t);
    h.up.set(0, 0, -1),
    h.lookAt(new n.Vector3(0,-1,0)),
    this.add(h);
    var c = new n.PerspectiveCamera(r,o,e,t);
    c.up.set(0, -1, 0),
    c.lookAt(new n.Vector3(0,0,1)),
    this.add(c);
    var u = new n.PerspectiveCamera(r,o,e,t);
    u.up.set(0, -1, 0),
    u.lookAt(new n.Vector3(0,0,-1)),
    this.add(u);
    var d = {
        format: n.RGBFormat,
        magFilter: n.LinearFilter,
        minFilter: n.LinearFilter
    };
    this.renderTarget = new n.WebGLRenderTargetCube(i,i,d),
    this.updateCubeMap = function(e, t) {
        null === this.parent && this.updateMatrixWorld();
        var i = this.renderTarget
          , n = i.texture.generateMipmaps;
        i.texture.generateMipmaps = !1,
        i.activeCubeFace = 0,
        e.render(t, a, i),
        i.activeCubeFace = 1,
        e.render(t, s, i),
        i.activeCubeFace = 2,
        e.render(t, l, i),
        i.activeCubeFace = 3,
        e.render(t, h, i),
        i.activeCubeFace = 4,
        e.render(t, c, i),
        i.texture.generateMipmaps = n,
        i.activeCubeFace = 5,
        e.render(t, u, i),
        e.setRenderTarget(null)
    }
}
,
n.CubeCamera.prototype = Object.create(n.Object3D.prototype),
n.CubeCamera.prototype.constructor = n.CubeCamera,
n.OrthographicCamera = function(e, t, i, r, o, a) {
    n.Camera.call(this),
    this.type = "OrthographicCamera",
    this.zoom = 1,
    this.left = e,
    this.right = t,
    this.top = i,
    this.bottom = r,
    this.near = void 0 !== o ? o : .1,
    this.far = void 0 !== a ? a : 2e3,
    this.updateProjectionMatrix()
}
,
n.OrthographicCamera.prototype = Object.create(n.Camera.prototype),
n.OrthographicCamera.prototype.constructor = n.OrthographicCamera,
n.OrthographicCamera.prototype.updateProjectionMatrix = function() {
    var e = (this.right - this.left) / (2 * this.zoom)
      , t = (this.top - this.bottom) / (2 * this.zoom)
      , i = (this.right + this.left) / 2
      , n = (this.top + this.bottom) / 2;
    this.projectionMatrix.makeOrthographic(i - e, i + e, n + t, n - t, this.near, this.far)
}
,
n.OrthographicCamera.prototype.copy = function(e) {
    return n.Camera.prototype.copy.call(this, e),
    this.left = e.left,
    this.right = e.right,
    this.top = e.top,
    this.bottom = e.bottom,
    this.near = e.near,
    this.far = e.far,
    this.zoom = e.zoom,
    this
}
,
n.OrthographicCamera.prototype.toJSON = function(e) {
    var t = n.Object3D.prototype.toJSON.call(this, e);
    return t.object.zoom = this.zoom,
    t.object.left = this.left,
    t.object.right = this.right,
    t.object.top = this.top,
    t.object.bottom = this.bottom,
    t.object.near = this.near,
    t.object.far = this.far,
    t
}
,
n.PerspectiveCamera = function(e, t, i, r) {
    n.Camera.call(this),
    this.type = "PerspectiveCamera",
    this.focalLength = 10,
    this.zoom = 1,
    this.fov = void 0 !== e ? e : 50,
    this.aspect = void 0 !== t ? t : 1,
    this.near = void 0 !== i ? i : .1,
    this.far = void 0 !== r ? r : 2e3,
    this.updateProjectionMatrix()
}
,
n.PerspectiveCamera.prototype = Object.create(n.Camera.prototype),
n.PerspectiveCamera.prototype.constructor = n.PerspectiveCamera,
n.PerspectiveCamera.prototype.setLens = function(e, t) {
    void 0 === t && (t = 24),
    this.fov = 2 * n.Math.radToDeg(Math.atan(t / (2 * e))),
    this.updateProjectionMatrix()
}
,
n.PerspectiveCamera.prototype.setViewOffset = function(e, t, i, n, r, o) {
    this.fullWidth = e,
    this.fullHeight = t,
    this.x = i,
    this.y = n,
    this.width = r,
    this.height = o,
    this.updateProjectionMatrix()
}
,
n.PerspectiveCamera.prototype.updateProjectionMatrix = function() {
    var e = n.Math.radToDeg(2 * Math.atan(Math.tan(.5 * n.Math.degToRad(this.fov)) / this.zoom));
    if (this.fullWidth) {
        var t = this.fullWidth / this.fullHeight
          , i = Math.tan(n.Math.degToRad(.5 * e)) * this.near
          , r = -i
          , o = t * r
          , a = t * i
          , s = Math.abs(a - o)
          , l = Math.abs(i - r);
        this.projectionMatrix.makeFrustum(o + this.x * s / this.fullWidth, o + (this.x + this.width) * s / this.fullWidth, i - (this.y + this.height) * l / this.fullHeight, i - this.y * l / this.fullHeight, this.near, this.far)
    } else
        this.projectionMatrix.makePerspective(e, this.aspect, this.near, this.far)
}
,
n.PerspectiveCamera.prototype.copy = function(e) {
    return n.Camera.prototype.copy.call(this, e),
    this.focalLength = e.focalLength,
    this.zoom = e.zoom,
    this.fov = e.fov,
    this.aspect = e.aspect,
    this.near = e.near,
    this.far = e.far,
    this
}
,
n.PerspectiveCamera.prototype.toJSON = function(e) {
    var t = n.Object3D.prototype.toJSON.call(this, e);
    return t.object.focalLength = this.focalLength,
    t.object.zoom = this.zoom,
    t.object.fov = this.fov,
    t.object.aspect = this.aspect,
    t.object.near = this.near,
    t.object.far = this.far,
    t
}
,
n.StereoCamera = function() {
    this.type = "StereoCamera",
    this.aspect = 1,
    this.cameraL = new n.PerspectiveCamera,
    this.cameraL.layers.enable(1),
    this.cameraL.matrixAutoUpdate = !1,
    this.cameraR = new n.PerspectiveCamera,
    this.cameraR.layers.enable(2),
    this.cameraR.matrixAutoUpdate = !1
}
,
n.StereoCamera.prototype = {
    constructor: n.StereoCamera,
    update: function() {
        var e, t, i, r, o, a = new n.Matrix4, s = new n.Matrix4;
        return function(l) {
            var h = e !== l.focalLength || t !== l.fov || i !== l.aspect * this.aspect || r !== l.near || o !== l.far;
            if (h) {
                e = l.focalLength,
                t = l.fov,
                i = l.aspect * this.aspect,
                r = l.near,
                o = l.far;
                var c, u, d = l.projectionMatrix.clone(), p = .032, f = p * r / e, g = r * Math.tan(n.Math.degToRad(.5 * t));
                s.elements[12] = -p,
                a.elements[12] = p,
                c = -g * i + f,
                u = g * i + f,
                d.elements[0] = 2 * r / (u - c),
                d.elements[8] = (u + c) / (u - c),
                this.cameraL.projectionMatrix.copy(d),
                c = -g * i - f,
                u = g * i - f,
                d.elements[0] = 2 * r / (u - c),
                d.elements[8] = (u + c) / (u - c),
                this.cameraR.projectionMatrix.copy(d)
            }
            this.cameraL.matrixWorld.copy(l.matrixWorld).multiply(s),
            this.cameraR.matrixWorld.copy(l.matrixWorld).multiply(a)
        }
    }()
},
n.Light = function(e, t) {
    n.Object3D.call(this),
    this.type = "Light",
    this.color = new n.Color(e),
    this.intensity = void 0 !== t ? t : 1,
    this.receiveShadow = void 0
}
,
n.Light.prototype = Object.create(n.Object3D.prototype),
n.Light.prototype.constructor = n.Light,
n.Light.prototype.copy = function(e) {
    return n.Object3D.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.intensity = e.intensity,
    this
}
,
n.Light.prototype.toJSON = function(e) {
    var t = n.Object3D.prototype.toJSON.call(this, e);
    return t.object.color = this.color.getHex(),
    t.object.intensity = this.intensity,
    void 0 !== this.groundColor && (t.object.groundColor = this.groundColor.getHex()),
    void 0 !== this.distance && (t.object.distance = this.distance),
    void 0 !== this.angle && (t.object.angle = this.angle),
    void 0 !== this.decay && (t.object.decay = this.decay),
    void 0 !== this.penumbra && (t.object.penumbra = this.penumbra),
    t
}
,
n.LightShadow = function(e) {
    this.camera = e,
    this.bias = 0,
    this.radius = 1,
    this.mapSize = new n.Vector2(512,512),
    this.map = null,
    this.matrix = new n.Matrix4
}
,
n.LightShadow.prototype = {
    constructor: n.LightShadow,
    copy: function(e) {
        return this.camera = e.camera.clone(),
        this.bias = e.bias,
        this.radius = e.radius,
        this.mapSize.copy(e.mapSize),
        this
    },
    clone: function() {
        return (new this.constructor).copy(this)
    }
},
n.AmbientLight = function(e, t) {
    n.Light.call(this, e, t),
    this.type = "AmbientLight",
    this.castShadow = void 0
}
,
n.AmbientLight.prototype = Object.create(n.Light.prototype),
n.AmbientLight.prototype.constructor = n.AmbientLight,
n.DirectionalLight = function(e, t) {
    n.Light.call(this, e, t),
    this.type = "DirectionalLight",
    this.position.set(0, 1, 0),
    this.updateMatrix(),
    this.target = new n.Object3D,
    this.shadow = new n.LightShadow(new n.OrthographicCamera(-5,5,5,-5,.5,500))
}
,
n.DirectionalLight.prototype = Object.create(n.Light.prototype),
n.DirectionalLight.prototype.constructor = n.DirectionalLight,
n.DirectionalLight.prototype.copy = function(e) {
    return n.Light.prototype.copy.call(this, e),
    this.target = e.target.clone(),
    this.shadow = e.shadow.clone(),
    this
}
,
n.HemisphereLight = function(e, t, i) {
    n.Light.call(this, e, i),
    this.type = "HemisphereLight",
    this.castShadow = void 0,
    this.position.set(0, 1, 0),
    this.updateMatrix(),
    this.groundColor = new n.Color(t)
}
,
n.HemisphereLight.prototype = Object.create(n.Light.prototype),
n.HemisphereLight.prototype.constructor = n.HemisphereLight,
n.HemisphereLight.prototype.copy = function(e) {
    return n.Light.prototype.copy.call(this, e),
    this.groundColor.copy(e.groundColor),
    this
}
,
n.PointLight = function(e, t, i, r) {
    n.Light.call(this, e, t),
    this.type = "PointLight",
    this.distance = void 0 !== i ? i : 0,
    this.decay = void 0 !== r ? r : 1,
    this.shadow = new n.LightShadow(new n.PerspectiveCamera(90,1,.5,500))
}
,
n.PointLight.prototype = Object.create(n.Light.prototype),
n.PointLight.prototype.constructor = n.PointLight,
Object.defineProperty(n.PointLight.prototype, "power", {
    get: function() {
        return 4 * this.intensity * Math.PI
    },
    set: function(e) {
        this.intensity = e / (4 * Math.PI)
    }
}),
n.PointLight.prototype.copy = function(e) {
    return n.Light.prototype.copy.call(this, e),
    this.distance = e.distance,
    this.decay = e.decay,
    this.shadow = e.shadow.clone(),
    this
}
,
n.SpotLight = function(e, t, i, r, o, a) {
    n.Light.call(this, e, t),
    this.type = "SpotLight",
    this.position.set(0, 1, 0),
    this.updateMatrix(),
    this.target = new n.Object3D,
    this.distance = void 0 !== i ? i : 0,
    this.angle = void 0 !== r ? r : Math.PI / 3,
    this.penumbra = void 0 !== o ? o : 0,
    this.decay = void 0 !== a ? a : 1,
    this.shadow = new n.LightShadow(new n.PerspectiveCamera(50,1,.5,500))
}
,
n.SpotLight.prototype = Object.create(n.Light.prototype),
n.SpotLight.prototype.constructor = n.SpotLight,
Object.defineProperty(n.SpotLight.prototype, "power", {
    get: function() {
        return this.intensity * Math.PI
    },
    set: function(e) {
        this.intensity = e / Math.PI
    }
}),
n.SpotLight.prototype.copy = function(e) {
    return n.Light.prototype.copy.call(this, e),
    this.distance = e.distance,
    this.angle = e.angle,
    this.penumbra = e.penumbra,
    this.decay = e.decay,
    this.target = e.target.clone(),
    this.shadow = e.shadow.clone(),
    this
}
,
n.Cache = {
    enabled: !1,
    files: {},
    add: function(e, t) {
        this.enabled !== !1 && (this.files[e] = t)
    },
    get: function(e) {
        if (this.enabled !== !1)
            return this.files[e]
    },
    remove: function(e) {
        delete this.files[e]
    },
    clear: function() {
        this.files = {}
    }
},
n.Loader = function() {
    this.onLoadStart = function() {}
    ,
    this.onLoadProgress = function() {}
    ,
    this.onLoadComplete = function() {}
}
,
n.Loader.prototype = {
    constructor: n.Loader,
    crossOrigin: void 0,
    extractUrlBase: function(e) {
        var t = e.split("/");
        return 1 === t.length ? "./" : (t.pop(),
        t.join("/") + "/")
    },
    initMaterials: function(e, t, i) {
        for (var n = [], r = 0; r < e.length; ++r)
            n[r] = this.createMaterial(e[r], t, i);
        return n
    },
    createMaterial: function() {
        var e, t, i;
        return function(r, o, a) {
            function s(e, i, r, s, h) {
                var c, u = o + e, d = n.Loader.Handlers.get(u);
                null !== d ? c = d.load(u) : (t.setCrossOrigin(a),
                c = t.load(u)),
                void 0 !== i && (c.repeat.fromArray(i),
                1 !== i[0] && (c.wrapS = n.RepeatWrapping),
                1 !== i[1] && (c.wrapT = n.RepeatWrapping)),
                void 0 !== r && c.offset.fromArray(r),
                void 0 !== s && ("repeat" === s[0] && (c.wrapS = n.RepeatWrapping),
                "mirror" === s[0] && (c.wrapS = n.MirroredRepeatWrapping),
                "repeat" === s[1] && (c.wrapT = n.RepeatWrapping),
                "mirror" === s[1] && (c.wrapT = n.MirroredRepeatWrapping)),
                void 0 !== h && (c.anisotropy = h);
                var p = n.Math.generateUUID();
                return l[p] = c,
                p
            }
            void 0 === e && (e = new n.Color),
            void 0 === t && (t = new n.TextureLoader),
            void 0 === i && (i = new n.MaterialLoader);
            var l = {}
              , h = {
                uuid: n.Math.generateUUID(),
                type: "MeshLambertMaterial"
            };
            for (var c in r) {
                var u = r[c];
                switch (c) {
                case "DbgColor":
                case "DbgIndex":
                case "opticalDensity":
                case "illumination":
                    break;
                case "DbgName":
                    h.name = u;
                    break;
                case "blending":
                    h.blending = n[u];
                    break;
                case "colorAmbient":
                case "mapAmbient":
                    console.warn("THREE.Loader.createMaterial:", c, "is no longer supported.");
                    break;
                case "colorDiffuse":
                    h.color = e.fromArray(u).getHex();
                    break;
                case "colorSpecular":
                    h.specular = e.fromArray(u).getHex();
                    break;
                case "colorEmissive":
                    h.emissive = e.fromArray(u).getHex();
                    break;
                case "specularCoef":
                    h.shininess = u;
                    break;
                case "shading":
                    "basic" === u.toLowerCase() && (h.type = "MeshBasicMaterial"),
                    "phong" === u.toLowerCase() && (h.type = "MeshPhongMaterial");
                    break;
                case "mapDiffuse":
                    h.map = s(u, r.mapDiffuseRepeat, r.mapDiffuseOffset, r.mapDiffuseWrap, r.mapDiffuseAnisotropy);
                    break;
                case "mapDiffuseRepeat":
                case "mapDiffuseOffset":
                case "mapDiffuseWrap":
                case "mapDiffuseAnisotropy":
                    break;
                case "mapLight":
                    h.lightMap = s(u, r.mapLightRepeat, r.mapLightOffset, r.mapLightWrap, r.mapLightAnisotropy);
                    break;
                case "mapLightRepeat":
                case "mapLightOffset":
                case "mapLightWrap":
                case "mapLightAnisotropy":
                    break;
                case "mapAO":
                    h.aoMap = s(u, r.mapAORepeat, r.mapAOOffset, r.mapAOWrap, r.mapAOAnisotropy);
                    break;
                case "mapAORepeat":
                case "mapAOOffset":
                case "mapAOWrap":
                case "mapAOAnisotropy":
                    break;
                case "mapBump":
                    h.bumpMap = s(u, r.mapBumpRepeat, r.mapBumpOffset, r.mapBumpWrap, r.mapBumpAnisotropy);
                    break;
                case "mapBumpScale":
                    h.bumpScale = u;
                    break;
                case "mapBumpRepeat":
                case "mapBumpOffset":
                case "mapBumpWrap":
                case "mapBumpAnisotropy":
                    break;
                case "mapNormal":
                    h.normalMap = s(u, r.mapNormalRepeat, r.mapNormalOffset, r.mapNormalWrap, r.mapNormalAnisotropy);
                    break;
                case "mapNormalFactor":
                    h.normalScale = [u, u];
                    break;
                case "mapNormalRepeat":
                case "mapNormalOffset":
                case "mapNormalWrap":
                case "mapNormalAnisotropy":
                    break;
                case "mapSpecular":
                    h.specularMap = s(u, r.mapSpecularRepeat, r.mapSpecularOffset, r.mapSpecularWrap, r.mapSpecularAnisotropy);
                    break;
                case "mapSpecularRepeat":
                case "mapSpecularOffset":
                case "mapSpecularWrap":
                case "mapSpecularAnisotropy":
                    break;
                case "mapAlpha":
                    h.alphaMap = s(u, r.mapAlphaRepeat, r.mapAlphaOffset, r.mapAlphaWrap, r.mapAlphaAnisotropy);
                    break;
                case "mapAlphaRepeat":
                case "mapAlphaOffset":
                case "mapAlphaWrap":
                case "mapAlphaAnisotropy":
                    break;
                case "flipSided":
                    h.side = n.BackSide;
                    break;
                case "doubleSided":
                    h.side = n.DoubleSide;
                    break;
                case "transparency":
                    console.warn("THREE.Loader.createMaterial: transparency has been renamed to opacity"),
                    h.opacity = u;
                    break;
                case "depthTest":
                case "depthWrite":
                case "colorWrite":
                case "opacity":
                case "reflectivity":
                case "transparent":
                case "visible":
                case "wireframe":
                    h[c] = u;
                    break;
                case "vertexColors":
                    u === !0 && (h.vertexColors = n.VertexColors),
                    "face" === u && (h.vertexColors = n.FaceColors);
                    break;
                default:
                    console.error("THREE.Loader.createMaterial: Unsupported", c, u)
                }
            }
            return "MeshBasicMaterial" === h.type && delete h.emissive,
            "MeshPhongMaterial" !== h.type && delete h.specular,
            h.opacity < 1 && (h.transparent = !0),
            i.setTextures(l),
            i.parse(h)
        }
    }()
},
n.Loader.Handlers = {
    handlers: [],
    add: function(e, t) {
        this.handlers.push(e, t)
    },
    get: function(e) {
        for (var t = this.handlers, i = 0, n = t.length; i < n; i += 2) {
            var r = t[i]
              , o = t[i + 1];
            if (r.test(e))
                return o;
        }
        return null
    }
},
n.XHRLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.XHRLoader.prototype = {
    constructor: n.XHRLoader,
    load: function(e, t, i, r) {
        void 0 !== this.path && (e = this.path + e);
        var o = this
          , a = n.Cache.get(e);
        if (void 0 !== a)
            return t && setTimeout(function() {
                t(a)
            }, 0),
            a;
        var s = new XMLHttpRequest;
        return s.overrideMimeType("text/plain"),
        s.open("GET", e, !0),
        s.addEventListener("load", function(i) {
            var a = i.target.response;
            n.Cache.add(e, a),
            200 === this.status ? (t && t(a),
            o.manager.itemEnd(e)) : 0 === this.status ? (console.warn("THREE.XHRLoader: HTTP Status 0 received."),
            t && t(a),
            o.manager.itemEnd(e)) : (r && r(i),
            o.manager.itemError(e))
        }, !1),
        void 0 !== i && s.addEventListener("progress", function(e) {
            i(e)
        }, !1),
        s.addEventListener("error", function(t) {
            r && r(t),
            o.manager.itemError(e)
        }, !1),
        void 0 !== this.responseType && (s.responseType = this.responseType),
        void 0 !== this.withCredentials && (s.withCredentials = this.withCredentials),
        s.send(null),
        o.manager.itemStart(e),
        s
    },
    setPath: function(e) {
        this.path = e
    },
    setResponseType: function(e) {
        this.responseType = e
    },
    setWithCredentials: function(e) {
        this.withCredentials = e
    }
},
n.FontLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.FontLoader.prototype = {
    constructor: n.FontLoader,
    load: function(e, t, i, r) {
        var o = new n.XHRLoader(this.manager);
        o.load(e, function(e) {
            t(new n.Font(JSON.parse(e.substring(65, e.length - 2))))
        }, i, r)
    }
},
n.ImageLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.ImageLoader.prototype = {
    constructor: n.ImageLoader,
    load: function(e, t, i, r) {
        void 0 !== this.path && (e = this.path + e);
        var o = this
          , a = n.Cache.get(e);
        if (void 0 !== a)
            return o.manager.itemStart(e),
            t ? setTimeout(function() {
                t(a),
                o.manager.itemEnd(e)
            }, 0) : o.manager.itemEnd(e),
            a;
        var s = document.createElement("img");
        return s.addEventListener("load", function(i) {
            n.Cache.add(e, this),
            t && t(this),
            o.manager.itemEnd(e)
        }, !1),
        void 0 !== i && s.addEventListener("progress", function(e) {
            i(e)
        }, !1),
        s.addEventListener("error", function(t) {
            r && r(t),
            o.manager.itemError(e)
        }, !1),
        void 0 !== this.crossOrigin && (s.crossOrigin = this.crossOrigin),
        o.manager.itemStart(e),
        s.src = e,
        s
    },
    setCrossOrigin: function(e) {
        this.crossOrigin = e
    },
    setPath: function(e) {
        this.path = e
    }
},
n.JSONLoader = function(e) {
    "boolean" == typeof e && (console.warn("THREE.JSONLoader: showStatus parameter has been removed from constructor."),
    e = void 0),
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager,
    this.withCredentials = !1
}
,
n.JSONLoader.prototype = {
    constructor: n.JSONLoader,
    get statusDomElement() {
        return void 0 === this._statusDomElement && (this._statusDomElement = document.createElement("div")),
        console.warn("THREE.JSONLoader: .statusDomElement has been removed."),
        this._statusDomElement
    },
    load: function(e, t, i, r) {
        var o = this
          , a = this.texturePath && "string" == typeof this.texturePath ? this.texturePath : n.Loader.prototype.extractUrlBase(e)
          , s = new n.XHRLoader(this.manager);
        s.setWithCredentials(this.withCredentials),
        s.load(e, function(i) {
            var n = JSON.parse(i)
              , r = n.metadata;
            if (void 0 !== r) {
                var s = r.type;
                if (void 0 !== s) {
                    if ("object" === s.toLowerCase())
                        return void console.error("THREE.JSONLoader: " + e + " should be loaded with THREE.ObjectLoader instead.");
                    if ("scene" === s.toLowerCase())
                        return void console.error("THREE.JSONLoader: " + e + " should be loaded with THREE.SceneLoader instead.")
                }
            }
            var l = o.parse(n, a);
            t(l.geometry, l.materials)
        }, i, r)
    },
    setTexturePath: function(e) {
        this.texturePath = e
    },
    parse: function(e, t) {
        function i(t) {
            function i(e, t) {
                return e & 1 << t
            }
            var r, o, a, l, h, c, u, d, p, f, g, m, v, A, y, C, I, b, w, E, x, T, M, S, _, P, R, L = e.faces, O = e.vertices, D = e.normals, F = e.colors, N = 0;
            if (void 0 !== e.uvs) {
                for (r = 0; r < e.uvs.length; r++)
                    e.uvs[r].length && N++;
                for (r = 0; r < N; r++)
                    s.faceVertexUvs[r] = []
            }
            for (l = 0,
            h = O.length; l < h; )
                b = new n.Vector3,
                b.x = O[l++] * t,
                b.y = O[l++] * t,
                b.z = O[l++] * t,
                s.vertices.push(b);
            for (l = 0,
            h = L.length; l < h; )
                if (f = L[l++],
                g = i(f, 0),
                m = i(f, 1),
                v = i(f, 3),
                A = i(f, 4),
                y = i(f, 5),
                C = i(f, 6),
                I = i(f, 7),
                g) {
                    if (E = new n.Face3,
                    E.a = L[l],
                    E.b = L[l + 1],
                    E.c = L[l + 3],
                    x = new n.Face3,
                    x.a = L[l + 1],
                    x.b = L[l + 2],
                    x.c = L[l + 3],
                    l += 4,
                    m && (p = L[l++],
                    E.materialIndex = p,
                    x.materialIndex = p),
                    a = s.faces.length,
                    v)
                        for (r = 0; r < N; r++)
                            for (S = e.uvs[r],
                            s.faceVertexUvs[r][a] = [],
                            s.faceVertexUvs[r][a + 1] = [],
                            o = 0; o < 4; o++)
                                d = L[l++],
                                P = S[2 * d],
                                R = S[2 * d + 1],
                                _ = new n.Vector2(P,R),
                                2 !== o && s.faceVertexUvs[r][a].push(_),
                                0 !== o && s.faceVertexUvs[r][a + 1].push(_);
                    if (A && (u = 3 * L[l++],
                    E.normal.set(D[u++], D[u++], D[u]),
                    x.normal.copy(E.normal)),
                    y)
                        for (r = 0; r < 4; r++)
                            u = 3 * L[l++],
                            M = new n.Vector3(D[u++],D[u++],D[u]),
                            2 !== r && E.vertexNormals.push(M),
                            0 !== r && x.vertexNormals.push(M);
                    if (C && (c = L[l++],
                    T = F[c],
                    E.color.setHex(T),
                    x.color.setHex(T)),
                    I)
                        for (r = 0; r < 4; r++)
                            c = L[l++],
                            T = F[c],
                            2 !== r && E.vertexColors.push(new n.Color(T)),
                            0 !== r && x.vertexColors.push(new n.Color(T));
                    s.faces.push(E),
                    s.faces.push(x)
                } else {
                    if (w = new n.Face3,
                    w.a = L[l++],
                    w.b = L[l++],
                    w.c = L[l++],
                    m && (p = L[l++],
                    w.materialIndex = p),
                    a = s.faces.length,
                    v)
                        for (r = 0; r < N; r++)
                            for (S = e.uvs[r],
                            s.faceVertexUvs[r][a] = [],
                            o = 0; o < 3; o++)
                                d = L[l++],
                                P = S[2 * d],
                                R = S[2 * d + 1],
                                _ = new n.Vector2(P,R),
                                s.faceVertexUvs[r][a].push(_);
                    if (A && (u = 3 * L[l++],
                    w.normal.set(D[u++], D[u++], D[u])),
                    y)
                        for (r = 0; r < 3; r++)
                            u = 3 * L[l++],
                            M = new n.Vector3(D[u++],D[u++],D[u]),
                            w.vertexNormals.push(M);
                    if (C && (c = L[l++],
                    w.color.setHex(F[c])),
                    I)
                        for (r = 0; r < 3; r++)
                            c = L[l++],
                            w.vertexColors.push(new n.Color(F[c]));
                    s.faces.push(w)
                }
        }
        function r() {
            var t = void 0 !== e.influencesPerVertex ? e.influencesPerVertex : 2;
            if (e.skinWeights)
                for (var i = 0, r = e.skinWeights.length; i < r; i += t) {
                    var o = e.skinWeights[i]
                      , a = t > 1 ? e.skinWeights[i + 1] : 0
                      , l = t > 2 ? e.skinWeights[i + 2] : 0
                      , h = t > 3 ? e.skinWeights[i + 3] : 0;
                    s.skinWeights.push(new n.Vector4(o,a,l,h))
                }
            if (e.skinIndices)
                for (var i = 0, r = e.skinIndices.length; i < r; i += t) {
                    var c = e.skinIndices[i]
                      , u = t > 1 ? e.skinIndices[i + 1] : 0
                      , d = t > 2 ? e.skinIndices[i + 2] : 0
                      , p = t > 3 ? e.skinIndices[i + 3] : 0;
                    s.skinIndices.push(new n.Vector4(c,u,d,p))
                }
            s.bones = e.bones,
            s.bones && s.bones.length > 0 && (s.skinWeights.length !== s.skinIndices.length || s.skinIndices.length !== s.vertices.length) && console.warn("When skinning, number of vertices (" + s.vertices.length + "), skinIndices (" + s.skinIndices.length + "), and skinWeights (" + s.skinWeights.length + ") should match.")
        }
        function o(t) {
            if (void 0 !== e.morphTargets)
                for (var i = 0, r = e.morphTargets.length; i < r; i++) {
                    s.morphTargets[i] = {},
                    s.morphTargets[i].name = e.morphTargets[i].name,
                    s.morphTargets[i].vertices = [];
                    for (var o = s.morphTargets[i].vertices, a = e.morphTargets[i].vertices, l = 0, h = a.length; l < h; l += 3) {
                        var c = new n.Vector3;
                        c.x = a[l] * t,
                        c.y = a[l + 1] * t,
                        c.z = a[l + 2] * t,
                        o.push(c)
                    }
                }
            if (void 0 !== e.morphColors && e.morphColors.length > 0) {
                console.warn('THREE.JSONLoader: "morphColors" no longer supported. Using them as face colors.');
                for (var u = s.faces, d = e.morphColors[0].colors, i = 0, r = u.length; i < r; i++)
                    u[i].color.fromArray(d, 3 * i)
            }
        }
        function a() {
            var t = []
              , i = [];
            void 0 !== e.animation && i.push(e.animation),
            void 0 !== e.animations && (e.animations.length ? i = i.concat(e.animations) : i.push(e.animations));
            for (var r = 0; r < i.length; r++) {
                var o = n.AnimationClip.parseAnimation(i[r], s.bones);
                o && t.push(o)
            }
            if (s.morphTargets) {
                var a = n.AnimationClip.CreateClipsFromMorphTargetSequences(s.morphTargets, 10);
                t = t.concat(a)
            }
            t.length > 0 && (s.animations = t)
        }
        var s = new n.Geometry
          , l = void 0 !== e.scale ? 1 / e.scale : 1;
        if (i(l),
        r(),
        o(l),
        a(),
        s.computeFaceNormals(),
        s.computeBoundingSphere(),
        void 0 === e.materials || 0 === e.materials.length)
            return {
                geometry: s
            };
        var h = n.Loader.prototype.initMaterials(e.materials, t, this.crossOrigin);
        return {
            geometry: s,
            materials: h
        }
    }
},
n.LoadingManager = function(e, t, i) {
    var n = this
      , r = !1
      , o = 0
      , a = 0;
    this.onStart = void 0,
    this.onLoad = e,
    this.onProgress = t,
    this.onError = i,
    this.itemStart = function(e) {
        a++,
        r === !1 && void 0 !== n.onStart && n.onStart(e, o, a),
        r = !0
    }
    ,
    this.itemEnd = function(e) {
        o++,
        void 0 !== n.onProgress && n.onProgress(e, o, a),
        o === a && (r = !1,
        void 0 !== n.onLoad && n.onLoad())
    }
    ,
    this.itemError = function(e) {
        void 0 !== n.onError && n.onError(e)
    }
}
,
n.DefaultLoadingManager = new n.LoadingManager,
n.BufferGeometryLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.BufferGeometryLoader.prototype = {
    constructor: n.BufferGeometryLoader,
    load: function(e, t, i, r) {
        var o = this
          , a = new n.XHRLoader(o.manager);
        a.load(e, function(e) {
            t(o.parse(JSON.parse(e)))
        }, i, r)
    },
    parse: function(e) {
        var t = new n.BufferGeometry
          , i = e.data.index
          , r = {
            Int8Array: Int8Array,
            Uint8Array: Uint8Array,
            Uint8ClampedArray: Uint8ClampedArray,
            Int16Array: Int16Array,
            Uint16Array: Uint16Array,
            Int32Array: Int32Array,
            Uint32Array: Uint32Array,
            Float32Array: Float32Array,
            Float64Array: Float64Array
        };
        if (void 0 !== i) {
            var o = new r[i.type](i.array);
            t.setIndex(new n.BufferAttribute(o,1))
        }
        var a = e.data.attributes;
        for (var s in a) {
            var l = a[s]
              , o = new r[l.type](l.array);
            t.addAttribute(s, new n.BufferAttribute(o,l.itemSize))
        }
        var h = e.data.groups || e.data.drawcalls || e.data.offsets;
        if (void 0 !== h)
            for (var c = 0, u = h.length; c !== u; ++c) {
                var d = h[c];
                t.addGroup(d.start, d.count, d.materialIndex)
            }
        var p = e.data.boundingSphere;
        if (void 0 !== p) {
            var f = new n.Vector3;
            void 0 !== p.center && f.fromArray(p.center),
            t.boundingSphere = new n.Sphere(f,p.radius)
        }
        return t
    }
},
n.MaterialLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager,
    this.textures = {}
}
,
n.MaterialLoader.prototype = {
    constructor: n.MaterialLoader,
    load: function(e, t, i, r) {
        var o = this
          , a = new n.XHRLoader(o.manager);
        a.load(e, function(e) {
            t(o.parse(JSON.parse(e)))
        }, i, r)
    },
    setTextures: function(e) {
        this.textures = e
    },
    getTexture: function(e) {
        var t = this.textures;
        return void 0 === t[e] && console.warn("THREE.MaterialLoader: Undefined texture", e),
        t[e]
    },
    parse: function(e) {
        var t = new n[e.type];
        if (void 0 !== e.uuid && (t.uuid = e.uuid),
        void 0 !== e.name && (t.name = e.name),
        void 0 !== e.color && t.color.setHex(e.color),
        void 0 !== e.roughness && (t.roughness = e.roughness),
        void 0 !== e.metalness && (t.metalness = e.metalness),
        void 0 !== e.emissive && t.emissive.setHex(e.emissive),
        void 0 !== e.specular && t.specular.setHex(e.specular),
        void 0 !== e.shininess && (t.shininess = e.shininess),
        void 0 !== e.uniforms && (t.uniforms = e.uniforms),
        void 0 !== e.vertexShader && (t.vertexShader = e.vertexShader),
        void 0 !== e.fragmentShader && (t.fragmentShader = e.fragmentShader),
        void 0 !== e.vertexColors && (t.vertexColors = e.vertexColors),
        void 0 !== e.shading && (t.shading = e.shading),
        void 0 !== e.blending && (t.blending = e.blending),
        void 0 !== e.side && (t.side = e.side),
        void 0 !== e.opacity && (t.opacity = e.opacity),
        void 0 !== e.transparent && (t.transparent = e.transparent),
        void 0 !== e.alphaTest && (t.alphaTest = e.alphaTest),
        void 0 !== e.depthTest && (t.depthTest = e.depthTest),
        void 0 !== e.depthWrite && (t.depthWrite = e.depthWrite),
        void 0 !== e.colorWrite && (t.colorWrite = e.colorWrite),
        void 0 !== e.wireframe && (t.wireframe = e.wireframe),
        void 0 !== e.wireframeLinewidth && (t.wireframeLinewidth = e.wireframeLinewidth),
        void 0 !== e.size && (t.size = e.size),
        void 0 !== e.sizeAttenuation && (t.sizeAttenuation = e.sizeAttenuation),
        void 0 !== e.map && (t.map = this.getTexture(e.map)),
        void 0 !== e.alphaMap && (t.alphaMap = this.getTexture(e.alphaMap),
        t.transparent = !0),
        void 0 !== e.bumpMap && (t.bumpMap = this.getTexture(e.bumpMap)),
        void 0 !== e.bumpScale && (t.bumpScale = e.bumpScale),
        void 0 !== e.normalMap && (t.normalMap = this.getTexture(e.normalMap)),
        void 0 !== e.normalScale) {
            var i = e.normalScale;
            Array.isArray(i) === !1 && (i = [i, i]),
            t.normalScale = (new n.Vector2).fromArray(i)
        }
        if (void 0 !== e.displacementMap && (t.displacementMap = this.getTexture(e.displacementMap)),
        void 0 !== e.displacementScale && (t.displacementScale = e.displacementScale),
        void 0 !== e.displacementBias && (t.displacementBias = e.displacementBias),
        void 0 !== e.roughnessMap && (t.roughnessMap = this.getTexture(e.roughnessMap)),
        void 0 !== e.metalnessMap && (t.metalnessMap = this.getTexture(e.metalnessMap)),
        void 0 !== e.emissiveMap && (t.emissiveMap = this.getTexture(e.emissiveMap)),
        void 0 !== e.emissiveIntensity && (t.emissiveIntensity = e.emissiveIntensity),
        void 0 !== e.specularMap && (t.specularMap = this.getTexture(e.specularMap)),
        void 0 !== e.envMap && (t.envMap = this.getTexture(e.envMap),
        t.combine = n.MultiplyOperation),
        e.reflectivity && (t.reflectivity = e.reflectivity),
        void 0 !== e.lightMap && (t.lightMap = this.getTexture(e.lightMap)),
        void 0 !== e.lightMapIntensity && (t.lightMapIntensity = e.lightMapIntensity),
        void 0 !== e.aoMap && (t.aoMap = this.getTexture(e.aoMap)),
        void 0 !== e.aoMapIntensity && (t.aoMapIntensity = e.aoMapIntensity),
        void 0 !== e.materials)
            for (var r = 0, o = e.materials.length; r < o; r++)
                t.materials.push(this.parse(e.materials[r]));
        return t
    }
},
n.ObjectLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager,
    this.texturePath = ""
}
,
n.ObjectLoader.prototype = {
    constructor: n.ObjectLoader,
    load: function(e, t, i, r) {
        "" === this.texturePath && (this.texturePath = e.substring(0, e.lastIndexOf("/") + 1));
        var o = this
          , a = new n.XHRLoader(o.manager);
        a.load(e, function(e) {
            o.parse(JSON.parse(e), t)
        }, i, r)
    },
    setTexturePath: function(e) {
        this.texturePath = e
    },
    setCrossOrigin: function(e) {
        this.crossOrigin = e
    },
    parse: function(e, t) {
        var i = this.parseGeometries(e.geometries)
          , n = this.parseImages(e.images, function() {
            void 0 !== t && t(a)
        })
          , r = this.parseTextures(e.textures, n)
          , o = this.parseMaterials(e.materials, r)
          , a = this.parseObject(e.object, i, o);
        return e.animations && (a.animations = this.parseAnimations(e.animations)),
        void 0 !== e.images && 0 !== e.images.length || void 0 !== t && t(a),
        a
    },
    parseGeometries: function(e) {
        var t = {};
        if (void 0 !== e)
            for (var i = new n.JSONLoader, r = new n.BufferGeometryLoader, o = 0, a = e.length; o < a; o++) {
                var s, l = e[o];
                switch (l.type) {
                case "PlaneGeometry":
                case "PlaneBufferGeometry":
                    s = new n[l.type](l.width,l.height,l.widthSegments,l.heightSegments);
                    break;
                case "BoxGeometry":
                case "BoxBufferGeometry":
                case "CubeGeometry":
                    s = new n[l.type](l.width,l.height,l.depth,l.widthSegments,l.heightSegments,l.depthSegments);
                    break;
                case "CircleGeometry":
                case "CircleBufferGeometry":
                    s = new n[l.type](l.radius,l.segments,l.thetaStart,l.thetaLength);
                    break;
                case "CylinderGeometry":
                case "CylinderBufferGeometry":
                    s = new n[l.type](l.radiusTop,l.radiusBottom,l.height,l.radialSegments,l.heightSegments,l.openEnded,l.thetaStart,l.thetaLength);
                    break;
                case "SphereGeometry":
                case "SphereBufferGeometry":
                    s = new n[l.type](l.radius,l.widthSegments,l.heightSegments,l.phiStart,l.phiLength,l.thetaStart,l.thetaLength);
                    break;
                case "DodecahedronGeometry":
                    s = new n.DodecahedronGeometry(l.radius,l.detail);
                    break;
                case "IcosahedronGeometry":
                    s = new n.IcosahedronGeometry(l.radius,l.detail);
                    break;
                case "OctahedronGeometry":
                    s = new n.OctahedronGeometry(l.radius,l.detail);
                    break;
                case "TetrahedronGeometry":
                    s = new n.TetrahedronGeometry(l.radius,l.detail);
                    break;
                case "RingGeometry":
                case "RingBufferGeometry":
                    s = new n[l.type](l.innerRadius,l.outerRadius,l.thetaSegments,l.phiSegments,l.thetaStart,l.thetaLength);
                    break;
                case "TorusGeometry":
                case "TorusBufferGeometry":
                    s = new n[l.type](l.radius,l.tube,l.radialSegments,l.tubularSegments,l.arc);
                    break;
                case "TorusKnotGeometry":
                case "TorusKnotBufferGeometry":
                    s = new n[l.type](l.radius,l.tube,l.tubularSegments,l.radialSegments,l.p,l.q);
                    break;
                case "LatheGeometry":
                    s = new n.LatheGeometry(l.points,l.segments,l.phiStart,l.phiLength);
                    break;
                case "BufferGeometry":
                    s = r.parse(l);
                    break;
                case "Geometry":
                    s = i.parse(l.data, this.texturePath).geometry;
                    break;
                default:
                    console.warn('THREE.ObjectLoader: Unsupported geometry type "' + l.type + '"');
                    continue
                }
                s.uuid = l.uuid,
                void 0 !== l.name && (s.name = l.name),
                t[l.uuid] = s
            }
        return t
    },
    parseMaterials: function(e, t) {
        var i = {};
        if (void 0 !== e) {
            var r = new n.MaterialLoader;
            r.setTextures(t);
            for (var o = 0, a = e.length; o < a; o++) {
                var s = r.parse(e[o]);
                i[s.uuid] = s
            }
        }
        return i
    },
    parseAnimations: function(e) {
        for (var t = [], i = 0; i < e.length; i++) {
            var r = n.AnimationClip.parse(e[i]);
            t.push(r)
        }
        return t
    },
    parseImages: function(e, t) {
        function i(e) {
            return r.manager.itemStart(e),
            s.load(e, function() {
                r.manager.itemEnd(e)
            })
        }
        var r = this
          , o = {};
        if (void 0 !== e && e.length > 0) {
            var a = new n.LoadingManager(t)
              , s = new n.ImageLoader(a);
            s.setCrossOrigin(this.crossOrigin);
            for (var l = 0, h = e.length; l < h; l++) {
                var c = e[l]
                  , u = /^(\/\/)|([a-z]+:(\/\/)?)/i.test(c.url) ? c.url : r.texturePath + c.url;
                o[c.uuid] = i(u)
            }
        }
        return o
    },
    parseTextures: function(e, t) {
        function i(e) {
            return "number" == typeof e ? e : (console.warn("THREE.ObjectLoader.parseTexture: Constant should be in numeric form.", e),
            n[e])
        }
        var r = {};
        if (void 0 !== e)
            for (var o = 0, a = e.length; o < a; o++) {
                var s = e[o];
                void 0 === s.image && console.warn('THREE.ObjectLoader: No "image" specified for', s.uuid),
                void 0 === t[s.image] && console.warn("THREE.ObjectLoader: Undefined image", s.image);
                var l = new n.Texture(t[s.image]);
                l.needsUpdate = !0,
                l.uuid = s.uuid,
                void 0 !== s.name && (l.name = s.name),
                void 0 !== s.mapping && (l.mapping = i(s.mapping)),
                void 0 !== s.offset && (l.offset = new n.Vector2(s.offset[0],s.offset[1])),
                void 0 !== s.repeat && (l.repeat = new n.Vector2(s.repeat[0],s.repeat[1])),
                void 0 !== s.minFilter && (l.minFilter = i(s.minFilter)),
                void 0 !== s.magFilter && (l.magFilter = i(s.magFilter)),
                void 0 !== s.anisotropy && (l.anisotropy = s.anisotropy),
                Array.isArray(s.wrap) && (l.wrapS = i(s.wrap[0]),
                l.wrapT = i(s.wrap[1])),
                r[s.uuid] = l
            }
        return r
    },
    parseObject: function() {
        var e = new n.Matrix4;
        return function(t, i, r) {
            function o(e) {
                return void 0 === i[e] && console.warn("THREE.ObjectLoader: Undefined geometry", e),
                i[e]
            }
            function a(e) {
                if (void 0 !== e)
                    return void 0 === r[e] && console.warn("THREE.ObjectLoader: Undefined material", e),
                    r[e]
            }
            var s;
            switch (t.type) {
            case "Scene":
                s = new n.Scene;
                break;
            case "PerspectiveCamera":
                s = new n.PerspectiveCamera(t.fov,t.aspect,t.near,t.far);
                break;
            case "OrthographicCamera":
                s = new n.OrthographicCamera(t.left,t.right,t.top,t.bottom,t.near,t.far);
                break;
            case "AmbientLight":
                s = new n.AmbientLight(t.color,t.intensity);
                break;
            case "DirectionalLight":
                s = new n.DirectionalLight(t.color,t.intensity);
                break;
            case "PointLight":
                s = new n.PointLight(t.color,t.intensity,t.distance,t.decay);
                break;
            case "SpotLight":
                s = new n.SpotLight(t.color,t.intensity,t.distance,t.angle,t.penumbra,t.decay);
                break;
            case "HemisphereLight":
                s = new n.HemisphereLight(t.color,t.groundColor,t.intensity);
                break;
            case "Mesh":
                var l = o(t.geometry)
                  , h = a(t.material);
                s = l.bones && l.bones.length > 0 ? new n.SkinnedMesh(l,h) : new n.Mesh(l,h);
                break;
            case "LOD":
                s = new n.LOD;
                break;
            case "Line":
                s = new n.Line(o(t.geometry),a(t.material),t.mode);
                break;
            case "PointCloud":
            case "Points":
                s = new n.Points(o(t.geometry),a(t.material));
                break;
            case "Sprite":
                s = new n.Sprite(a(t.material));
                break;
            case "Group":
                s = new n.Group;
                break;
            default:
                s = new n.Object3D
            }
            if (s.uuid = t.uuid,
            void 0 !== t.name && (s.name = t.name),
            void 0 !== t.matrix ? (e.fromArray(t.matrix),
            e.decompose(s.position, s.quaternion, s.scale)) : (void 0 !== t.position && s.position.fromArray(t.position),
            void 0 !== t.rotation && s.rotation.fromArray(t.rotation),
            void 0 !== t.scale && s.scale.fromArray(t.scale)),
            void 0 !== t.castShadow && (s.castShadow = t.castShadow),
            void 0 !== t.receiveShadow && (s.receiveShadow = t.receiveShadow),
            void 0 !== t.visible && (s.visible = t.visible),
            void 0 !== t.userData && (s.userData = t.userData),
            void 0 !== t.children)
                for (var c in t.children)
                    s.add(this.parseObject(t.children[c], i, r));
            if ("LOD" === t.type)
                for (var u = t.levels, d = 0; d < u.length; d++) {
                    var p = u[d]
                      , c = s.getObjectByProperty("uuid", p.object);
                    void 0 !== c && s.addLevel(c, p.distance)
                }
            return s
        }
    }()
},
n.TextureLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.TextureLoader.prototype = {
    constructor: n.TextureLoader,
    load: function(e, t, i, r) {
        var o = new n.Texture
          , a = new n.ImageLoader(this.manager);
        return a.setCrossOrigin(this.crossOrigin),
        a.setPath(this.path),
        a.load(e, function(e) {
            o.image = e,
            o.needsUpdate = !0,
            void 0 !== t && t(o)
        }, i, r),
        o
    },
    setCrossOrigin: function(e) {
        this.crossOrigin = e
    },
    setPath: function(e) {
        this.path = e
    }
},
n.CubeTextureLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager
}
,
n.CubeTextureLoader.prototype = {
    constructor: n.CubeTextureLoader,
    load: function(e, t, i, r) {
        function o(i) {
            s.load(e[i], function(e) {
                a.images[i] = e,
                l++,
                6 === l && (a.needsUpdate = !0,
                t && t(a))
            }, void 0, r)
        }
        var a = new n.CubeTexture
          , s = new n.ImageLoader(this.manager);
        s.setCrossOrigin(this.crossOrigin),
        s.setPath(this.path);
        for (var l = 0, h = 0; h < e.length; ++h)
            o(h);
        return a
    },
    setCrossOrigin: function(e) {
        this.crossOrigin = e
    },
    setPath: function(e) {
        this.path = e
    }
},
n.DataTextureLoader = n.BinaryTextureLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager,
    this._parser = null
}
,
n.BinaryTextureLoader.prototype = {
    constructor: n.BinaryTextureLoader,
    load: function(e, t, i, r) {
        var o = this
          , a = new n.DataTexture
          , s = new n.XHRLoader(this.manager);
        return s.setResponseType("arraybuffer"),
        s.load(e, function(e) {
            var i = o._parser(e);
            i && (void 0 !== i.image ? a.image = i.image : void 0 !== i.data && (a.image.width = i.width,
            a.image.height = i.height,
            a.image.data = i.data),
            a.wrapS = void 0 !== i.wrapS ? i.wrapS : n.ClampToEdgeWrapping,
            a.wrapT = void 0 !== i.wrapT ? i.wrapT : n.ClampToEdgeWrapping,
            a.magFilter = void 0 !== i.magFilter ? i.magFilter : n.LinearFilter,
            a.minFilter = void 0 !== i.minFilter ? i.minFilter : n.LinearMipMapLinearFilter,
            a.anisotropy = void 0 !== i.anisotropy ? i.anisotropy : 1,
            void 0 !== i.format && (a.format = i.format),
            void 0 !== i.type && (a.type = i.type),
            void 0 !== i.mipmaps && (a.mipmaps = i.mipmaps),
            1 === i.mipmapCount && (a.minFilter = n.LinearFilter),
            a.needsUpdate = !0,
            t && t(a, i))
        }, i, r),
        a
    }
},
n.CompressedTextureLoader = function(e) {
    this.manager = void 0 !== e ? e : n.DefaultLoadingManager,
    this._parser = null
}
;
n.CompressedTextureLoader.prototype = {
    constructor: n.CompressedTextureLoader,
    load: function(e, t, i, r) {
        function o(o) {
            h.load(e[o], function(e) {
                var i = a._parser(e, !0);
                s[o] = {
                    width: i.width,
                    height: i.height,
                    format: i.format,
                    mipmaps: i.mipmaps
                },
                c += 1,
                6 === c && (1 === i.mipmapCount && (l.minFilter = n.LinearFilter),
                l.format = i.format,
                l.needsUpdate = !0,
                t && t(l))
            }, i, r)
        }
        var a = this
          , s = []
          , l = new n.CompressedTexture;
        l.image = s;
        var h = new n.XHRLoader(this.manager);
        if (h.setPath(this.path),
        h.setResponseType("arraybuffer"),
        Array.isArray(e))
            for (var c = 0, u = 0, d = e.length; u < d; ++u)
                o(u);
        else
            h.load(e, function(e) {
                var i = a._parser(e, !0);
                if (i.isCubemap)
                    for (var r = i.mipmaps.length / i.mipmapCount, o = 0; o < r; o++) {
                        s[o] = {
                            mipmaps: []
                        };
                        for (var h = 0; h < i.mipmapCount; h++)
                            s[o].mipmaps.push(i.mipmaps[o * i.mipmapCount + h]),
                            s[o].format = i.format,
                            s[o].width = i.width,
                            s[o].height = i.height
                    }
                else
                    l.image.width = i.width,
                    l.image.height = i.height,
                    l.mipmaps = i.mipmaps;
                1 === i.mipmapCount && (l.minFilter = n.LinearFilter),
                l.format = i.format,
                l.needsUpdate = !0,
                t && t(l)
            }, i, r);
        return l
    },
    setPath: function(e) {
        this.path = e
    }
};
n.Material = function() {
    Object.defineProperty(this, "id", {
        value: n.MaterialIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.type = "Material",
    this.side = n.FrontSide,
    this.opacity = 1,
    this.transparent = !1,
    this.blending = n.NormalBlending,
    this.blendSrc = n.SrcAlphaFactor,
    this.blendDst = n.OneMinusSrcAlphaFactor,
    this.blendEquation = n.AddEquation,
    this.blendSrcAlpha = null,
    this.blendDstAlpha = null,
    this.blendEquationAlpha = null,
    this.depthFunc = n.LessEqualDepth,
    this.depthTest = !0,
    this.depthWrite = !0,
    this.colorWrite = !0,
    this.precision = null,
    this.polygonOffset = !1,
    this.polygonOffsetFactor = 0,
    this.polygonOffsetUnits = 0,
    this.alphaTest = 0,
    this.premultipliedAlpha = !1,
    this.overdraw = 0,
    this.visible = !0,
    this._needsUpdate = !0
}
,
n.Material.prototype = {
    constructor: n.Material,
    get needsUpdate() {
        return this._needsUpdate
    },
    set needsUpdate(e) {
        e === !0 && this.update(),
        this._needsUpdate = e
    },
    setValues: function(e) {
        if (void 0 !== e)
            for (var t in e) {
                var i = e[t];
                if (void 0 !== i) {
                    var r = this[t];
                    void 0 !== r ? r instanceof n.Color ? r.set(i) : r instanceof n.Vector3 && i instanceof n.Vector3 ? r.copy(i) : "overdraw" === t ? this[t] = Number(i) : this[t] = i : console.warn("THREE." + this.type + ": '" + t + "' is not a property of this material.")
                } else
                    console.warn("THREE.Material: '" + t + "' parameter is undefined.")
            }
    },
    toJSON: function(e) {
        function t(e) {
            var t = [];
            for (var i in e) {
                var n = e[i];
                delete n.metadata,
                t.push(n)
            }
            return t
        }
        var i = void 0 === e;
        i && (e = {
            textures: {},
            images: {}
        });
        var r = {
            metadata: {
                version: 4.4,
                type: "Material",
                generator: "Material.toJSON"
            }
        };
        if (r.uuid = this.uuid,
        r.type = this.type,
        "" !== this.name && (r.name = this.name),
        this.color instanceof n.Color && (r.color = this.color.getHex()),
        .5 !== this.roughness && (r.roughness = this.roughness),
        .5 !== this.metalness && (r.metalness = this.metalness),
        this.emissive instanceof n.Color && (r.emissive = this.emissive.getHex()),
        this.specular instanceof n.Color && (r.specular = this.specular.getHex()),
        void 0 !== this.shininess && (r.shininess = this.shininess),
        this.map instanceof n.Texture && (r.map = this.map.toJSON(e).uuid),
        this.alphaMap instanceof n.Texture && (r.alphaMap = this.alphaMap.toJSON(e).uuid),
        this.lightMap instanceof n.Texture && (r.lightMap = this.lightMap.toJSON(e).uuid),
        this.bumpMap instanceof n.Texture && (r.bumpMap = this.bumpMap.toJSON(e).uuid,
        r.bumpScale = this.bumpScale),
        this.normalMap instanceof n.Texture && (r.normalMap = this.normalMap.toJSON(e).uuid,
        r.normalScale = this.normalScale.toArray()),
        this.displacementMap instanceof n.Texture && (r.displacementMap = this.displacementMap.toJSON(e).uuid,
        r.displacementScale = this.displacementScale,
        r.displacementBias = this.displacementBias),
        this.roughnessMap instanceof n.Texture && (r.roughnessMap = this.roughnessMap.toJSON(e).uuid),
        this.metalnessMap instanceof n.Texture && (r.metalnessMap = this.metalnessMap.toJSON(e).uuid),
        this.emissiveMap instanceof n.Texture && (r.emissiveMap = this.emissiveMap.toJSON(e).uuid),
        this.specularMap instanceof n.Texture && (r.specularMap = this.specularMap.toJSON(e).uuid),
        this.envMap instanceof n.Texture && (r.envMap = this.envMap.toJSON(e).uuid,
        r.reflectivity = this.reflectivity),
        void 0 !== this.size && (r.size = this.size),
        void 0 !== this.sizeAttenuation && (r.sizeAttenuation = this.sizeAttenuation),
        void 0 !== this.vertexColors && this.vertexColors !== n.NoColors && (r.vertexColors = this.vertexColors),
        void 0 !== this.shading && this.shading !== n.SmoothShading && (r.shading = this.shading),
        void 0 !== this.blending && this.blending !== n.NormalBlending && (r.blending = this.blending),
        void 0 !== this.side && this.side !== n.FrontSide && (r.side = this.side),
        this.opacity < 1 && (r.opacity = this.opacity),
        this.transparent === !0 && (r.transparent = this.transparent),
        this.alphaTest > 0 && (r.alphaTest = this.alphaTest),
        this.premultipliedAlpha === !0 && (r.premultipliedAlpha = this.premultipliedAlpha),
        this.wireframe === !0 && (r.wireframe = this.wireframe),
        this.wireframeLinewidth > 1 && (r.wireframeLinewidth = this.wireframeLinewidth),
        i) {
            var o = t(e.textures)
              , a = t(e.images);
            o.length > 0 && (r.textures = o),
            a.length > 0 && (r.images = a)
        }
        return r
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.name = e.name,
        this.side = e.side,
        this.opacity = e.opacity,
        this.transparent = e.transparent,
        this.blending = e.blending,
        this.blendSrc = e.blendSrc,
        this.blendDst = e.blendDst,
        this.blendEquation = e.blendEquation,
        this.blendSrcAlpha = e.blendSrcAlpha,
        this.blendDstAlpha = e.blendDstAlpha,
        this.blendEquationAlpha = e.blendEquationAlpha,
        this.depthFunc = e.depthFunc,
        this.depthTest = e.depthTest,
        this.depthWrite = e.depthWrite,
        this.colorWrite = e.colorWrite,
        this.precision = e.precision,
        this.polygonOffset = e.polygonOffset,
        this.polygonOffsetFactor = e.polygonOffsetFactor,
        this.polygonOffsetUnits = e.polygonOffsetUnits,
        this.alphaTest = e.alphaTest,
        this.premultipliedAlpha = e.premultipliedAlpha,
        this.overdraw = e.overdraw,
        this.visible = e.visible,
        this
    },
    update: function() {
        this.dispatchEvent({
            type: "update"
        })
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    }
},
n.EventDispatcher.prototype.apply(n.Material.prototype),
n.MaterialIdCount = 0,
n.LineBasicMaterial = function(e) {
    n.Material.call(this),
    this.type = "LineBasicMaterial",
    this.color = new n.Color(16777215),
    this.linewidth = 1,
    this.linecap = "round",
    this.linejoin = "round",
    this.blending = n.NormalBlending,
    this.vertexColors = n.NoColors,
    this.fog = !0,
    this.setValues(e)
}
,
n.LineBasicMaterial.prototype = Object.create(n.Material.prototype),
n.LineBasicMaterial.prototype.constructor = n.LineBasicMaterial,
n.LineBasicMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.linewidth = e.linewidth,
    this.linecap = e.linecap,
    this.linejoin = e.linejoin,
    this.vertexColors = e.vertexColors,
    this.fog = e.fog,
    this
}
,
n.LineDashedMaterial = function(e) {
    n.Material.call(this),
    this.type = "LineDashedMaterial",
    this.color = new n.Color(16777215),
    this.linewidth = 1,
    this.scale = 1,
    this.dashSize = 3,
    this.gapSize = 1,
    this.blending = n.NormalBlending,
    this.vertexColors = n.NoColors,
    this.fog = !0,
    this.setValues(e)
}
,
n.LineDashedMaterial.prototype = Object.create(n.Material.prototype),
n.LineDashedMaterial.prototype.constructor = n.LineDashedMaterial,
n.LineDashedMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.linewidth = e.linewidth,
    this.scale = e.scale,
    this.dashSize = e.dashSize,
    this.gapSize = e.gapSize,
    this.vertexColors = e.vertexColors,
    this.fog = e.fog,
    this
}
,
n.MeshBasicMaterial = function(e) {
    n.Material.call(this),
    this.type = "MeshBasicMaterial",
    this.color = new n.Color(16777215),
    this.map = null,
    this.aoMap = null,
    this.aoMapIntensity = 1,
    this.specularMap = null,
    this.alphaMap = null,
    this.envMap = null,
    this.combine = n.MultiplyOperation,
    this.reflectivity = 1,
    this.refractionRatio = .98,
    this.fog = !0,
    this.shading = n.SmoothShading,
    this.blending = n.NormalBlending,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.wireframeLinecap = "round",
    this.wireframeLinejoin = "round",
    this.vertexColors = n.NoColors,
    this.skinning = !1,
    this.morphTargets = !1,
    this.setValues(e)
}
,
n.MeshBasicMaterial.prototype = Object.create(n.Material.prototype),
n.MeshBasicMaterial.prototype.constructor = n.MeshBasicMaterial,
n.MeshBasicMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.map = e.map,
    this.aoMap = e.aoMap,
    this.aoMapIntensity = e.aoMapIntensity,
    this.specularMap = e.specularMap,
    this.alphaMap = e.alphaMap,
    this.envMap = e.envMap,
    this.combine = e.combine,
    this.reflectivity = e.reflectivity,
    this.refractionRatio = e.refractionRatio,
    this.fog = e.fog,
    this.shading = e.shading,
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this.wireframeLinecap = e.wireframeLinecap,
    this.wireframeLinejoin = e.wireframeLinejoin,
    this.vertexColors = e.vertexColors,
    this.skinning = e.skinning,
    this.morphTargets = e.morphTargets,
    this
}
,
n.MeshLambertMaterial = function(e) {
    n.Material.call(this),
    this.type = "MeshLambertMaterial",
    this.color = new n.Color(16777215),
    this.map = null,
    this.lightMap = null,
    this.lightMapIntensity = 1,
    this.aoMap = null,
    this.aoMapIntensity = 1,
    this.emissive = new n.Color(0),
    this.emissiveIntensity = 1,
    this.emissiveMap = null,
    this.specularMap = null,
    this.alphaMap = null,
    this.envMap = null,
    this.combine = n.MultiplyOperation,
    this.reflectivity = 1,
    this.refractionRatio = .98,
    this.fog = !0,
    this.blending = n.NormalBlending,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.wireframeLinecap = "round",
    this.wireframeLinejoin = "round",
    this.vertexColors = n.NoColors,
    this.skinning = !1,
    this.morphTargets = !1,
    this.morphNormals = !1,
    this.setValues(e)
}
,
n.MeshLambertMaterial.prototype = Object.create(n.Material.prototype),
n.MeshLambertMaterial.prototype.constructor = n.MeshLambertMaterial,
n.MeshLambertMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.map = e.map,
    this.lightMap = e.lightMap,
    this.lightMapIntensity = e.lightMapIntensity,
    this.aoMap = e.aoMap,
    this.aoMapIntensity = e.aoMapIntensity,
    this.emissive.copy(e.emissive),
    this.emissiveMap = e.emissiveMap,
    this.emissiveIntensity = e.emissiveIntensity,
    this.specularMap = e.specularMap,
    this.alphaMap = e.alphaMap,
    this.envMap = e.envMap,
    this.combine = e.combine,
    this.reflectivity = e.reflectivity,
    this.refractionRatio = e.refractionRatio,
    this.fog = e.fog,
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this.wireframeLinecap = e.wireframeLinecap,
    this.wireframeLinejoin = e.wireframeLinejoin,
    this.vertexColors = e.vertexColors,
    this.skinning = e.skinning,
    this.morphTargets = e.morphTargets,
    this.morphNormals = e.morphNormals,
    this
}
,
n.MeshPhongMaterial = function(e) {
    n.Material.call(this),
    this.type = "MeshPhongMaterial",
    this.color = new n.Color(16777215),
    this.specular = new n.Color(1118481),
    this.shininess = 30,
    this.map = null,
    this.lightMap = null,
    this.lightMapIntensity = 1,
    this.aoMap = null,
    this.aoMapIntensity = 1,
    this.emissive = new n.Color(0),
    this.emissiveIntensity = 1,
    this.emissiveMap = null,
    this.bumpMap = null,
    this.bumpScale = 1,
    this.normalMap = null,
    this.normalScale = new n.Vector2(1,1),
    this.displacementMap = null,
    this.displacementScale = 1,
    this.displacementBias = 0,
    this.specularMap = null,
    this.alphaMap = null,
    this.envMap = null,
    this.combine = n.MultiplyOperation,
    this.reflectivity = 1,
    this.refractionRatio = .98,
    this.fog = !0,
    this.shading = n.SmoothShading,
    this.blending = n.NormalBlending,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.wireframeLinecap = "round",
    this.wireframeLinejoin = "round",
    this.vertexColors = n.NoColors,
    this.skinning = !1,
    this.morphTargets = !1,
    this.morphNormals = !1,
    this.setValues(e)
}
,
n.MeshPhongMaterial.prototype = Object.create(n.Material.prototype),
n.MeshPhongMaterial.prototype.constructor = n.MeshPhongMaterial,
n.MeshPhongMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.specular.copy(e.specular),
    this.shininess = e.shininess,
    this.map = e.map,
    this.lightMap = e.lightMap,
    this.lightMapIntensity = e.lightMapIntensity,
    this.aoMap = e.aoMap,
    this.aoMapIntensity = e.aoMapIntensity,
    this.emissive.copy(e.emissive),
    this.emissiveMap = e.emissiveMap,
    this.emissiveIntensity = e.emissiveIntensity,
    this.bumpMap = e.bumpMap,
    this.bumpScale = e.bumpScale,
    this.normalMap = e.normalMap,
    this.normalScale.copy(e.normalScale),
    this.displacementMap = e.displacementMap,
    this.displacementScale = e.displacementScale,
    this.displacementBias = e.displacementBias,
    this.specularMap = e.specularMap,
    this.alphaMap = e.alphaMap,
    this.envMap = e.envMap,
    this.combine = e.combine,
    this.reflectivity = e.reflectivity,
    this.refractionRatio = e.refractionRatio,
    this.fog = e.fog,
    this.shading = e.shading,
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this.wireframeLinecap = e.wireframeLinecap,
    this.wireframeLinejoin = e.wireframeLinejoin,
    this.vertexColors = e.vertexColors,
    this.skinning = e.skinning,
    this.morphTargets = e.morphTargets,
    this.morphNormals = e.morphNormals,
    this
}
,
n.MeshStandardMaterial = function(e) {
    n.Material.call(this),
    this.type = "MeshStandardMaterial",
    this.color = new n.Color(16777215),
    this.roughness = .5,
    this.metalness = .5,
    this.map = null,
    this.lightMap = null,
    this.lightMapIntensity = 1,
    this.aoMap = null,
    this.aoMapIntensity = 1,
    this.emissive = new n.Color(0),
    this.emissiveIntensity = 1,
    this.emissiveMap = null,
    this.bumpMap = null,
    this.bumpScale = 1,
    this.normalMap = null,
    this.normalScale = new n.Vector2(1,1),
    this.displacementMap = null,
    this.displacementScale = 1,
    this.displacementBias = 0,
    this.roughnessMap = null,
    this.metalnessMap = null,
    this.alphaMap = null,
    this.envMap = null,
    this.envMapIntensity = 1,
    this.refractionRatio = .98,
    this.fog = !0,
    this.shading = n.SmoothShading,
    this.blending = n.NormalBlending,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.wireframeLinecap = "round",
    this.wireframeLinejoin = "round",
    this.vertexColors = n.NoColors,
    this.skinning = !1,
    this.morphTargets = !1,
    this.morphNormals = !1,
    this.setValues(e)
}
,
n.MeshStandardMaterial.prototype = Object.create(n.Material.prototype),
n.MeshStandardMaterial.prototype.constructor = n.MeshStandardMaterial,
n.MeshStandardMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.roughness = e.roughness,
    this.metalness = e.metalness,
    this.map = e.map,
    this.lightMap = e.lightMap,
    this.lightMapIntensity = e.lightMapIntensity,
    this.aoMap = e.aoMap,
    this.aoMapIntensity = e.aoMapIntensity,
    this.emissive.copy(e.emissive),
    this.emissiveMap = e.emissiveMap,
    this.emissiveIntensity = e.emissiveIntensity,
    this.bumpMap = e.bumpMap,
    this.bumpScale = e.bumpScale,
    this.normalMap = e.normalMap,
    this.normalScale.copy(e.normalScale),
    this.displacementMap = e.displacementMap,
    this.displacementScale = e.displacementScale,
    this.displacementBias = e.displacementBias,
    this.roughnessMap = e.roughnessMap,
    this.metalnessMap = e.metalnessMap,
    this.alphaMap = e.alphaMap,
    this.envMap = e.envMap,
    this.envMapIntensity = e.envMapIntensity,
    this.refractionRatio = e.refractionRatio,
    this.fog = e.fog,
    this.shading = e.shading,
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this.wireframeLinecap = e.wireframeLinecap,
    this.wireframeLinejoin = e.wireframeLinejoin,
    this.vertexColors = e.vertexColors,
    this.skinning = e.skinning,
    this.morphTargets = e.morphTargets,
    this.morphNormals = e.morphNormals,
    this
}
,
n.MeshDepthMaterial = function(e) {
    n.Material.call(this),
    this.type = "MeshDepthMaterial",
    this.morphTargets = !1,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.setValues(e)
}
,
n.MeshDepthMaterial.prototype = Object.create(n.Material.prototype),
n.MeshDepthMaterial.prototype.constructor = n.MeshDepthMaterial,
n.MeshDepthMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this
}
,
n.MeshNormalMaterial = function(e) {
    n.Material.call(this, e),
    this.type = "MeshNormalMaterial",
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.morphTargets = !1,
    this.setValues(e)
}
,
n.MeshNormalMaterial.prototype = Object.create(n.Material.prototype),
n.MeshNormalMaterial.prototype.constructor = n.MeshNormalMaterial,
n.MeshNormalMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this
}
,
n.MultiMaterial = function(e) {
    this.uuid = n.Math.generateUUID(),
    this.type = "MultiMaterial",
    this.materials = e instanceof Array ? e : [],
    this.visible = !0
}
,
n.MultiMaterial.prototype = {
    constructor: n.MultiMaterial,
    toJSON: function(e) {
        for (var t = {
            metadata: {
                version: 4.2,
                type: "material",
                generator: "MaterialExporter"
            },
            uuid: this.uuid,
            type: this.type,
            materials: []
        }, i = this.materials, n = 0, r = i.length; n < r; n++) {
            var o = i[n].toJSON(e);
            delete o.metadata,
            t.materials.push(o)
        }
        return t.visible = this.visible,
        t
    },
    clone: function() {
        for (var e = new this.constructor, t = 0; t < this.materials.length; t++)
            e.materials.push(this.materials[t].clone());
        return e.visible = this.visible,
        e
    }
},
n.PointsMaterial = function(e) {
    n.Material.call(this),
    this.type = "PointsMaterial",
    this.color = new n.Color(16777215),
    this.map = null,
    this.size = 1,
    this.sizeAttenuation = !0,
    this.blending = n.NormalBlending,
    this.vertexColors = n.NoColors,
    this.fog = !0,
    this.setValues(e)
}
,
n.PointsMaterial.prototype = Object.create(n.Material.prototype),
n.PointsMaterial.prototype.constructor = n.PointsMaterial,
n.PointsMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.map = e.map,
    this.size = e.size,
    this.sizeAttenuation = e.sizeAttenuation,
    this.vertexColors = e.vertexColors,
    this.fog = e.fog,
    this
}
,
n.ShaderMaterial = function(e) {
    n.Material.call(this),
    this.type = "ShaderMaterial",
    this.defines = {},
    this.uniforms = {},
    this.vertexShader = "void main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}",
    this.fragmentShader = "void main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}",
    this.shading = n.SmoothShading,
    this.linewidth = 1,
    this.wireframe = !1,
    this.wireframeLinewidth = 1,
    this.fog = !1,
    this.lights = !1,
    this.vertexColors = n.NoColors,
    this.skinning = !1,
    this.morphTargets = !1,
    this.morphNormals = !1,
    this.extensions = {
        derivatives: !1,
        fragDepth: !1,
        drawBuffers: !1,
        shaderTextureLOD: !1
    },
    this.defaultAttributeValues = {
        color: [1, 1, 1],
        uv: [0, 0],
        uv2: [0, 0]
    },
    this.index0AttributeName = void 0,
    void 0 !== e && (void 0 !== e.attributes && console.error("THREE.ShaderMaterial: attributes should now be defined in THREE.BufferGeometry instead."),
    this.setValues(e))
}
,
n.ShaderMaterial.prototype = Object.create(n.Material.prototype),
n.ShaderMaterial.prototype.constructor = n.ShaderMaterial,
n.ShaderMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.fragmentShader = e.fragmentShader,
    this.vertexShader = e.vertexShader,
    this.uniforms = n.UniformsUtils.clone(e.uniforms),
    this.defines = e.defines,
    this.shading = e.shading,
    this.wireframe = e.wireframe,
    this.wireframeLinewidth = e.wireframeLinewidth,
    this.fog = e.fog,
    this.lights = e.lights,
    this.vertexColors = e.vertexColors,
    this.skinning = e.skinning,
    this.morphTargets = e.morphTargets,
    this.morphNormals = e.morphNormals,
    this.extensions = e.extensions,
    this
}
,
n.ShaderMaterial.prototype.toJSON = function(e) {
    var t = n.Material.prototype.toJSON.call(this, e);
    return t.uniforms = this.uniforms,
    t.vertexShader = this.vertexShader,
    t.fragmentShader = this.fragmentShader,
    t
}
,
n.RawShaderMaterial = function(e) {
    n.ShaderMaterial.call(this, e),
    this.type = "RawShaderMaterial"
}
,
n.RawShaderMaterial.prototype = Object.create(n.ShaderMaterial.prototype),
n.RawShaderMaterial.prototype.constructor = n.RawShaderMaterial,
n.SpriteMaterial = function(e) {
    n.Material.call(this),
    this.type = "SpriteMaterial",
    this.color = new n.Color(16777215),
    this.map = null,
    this.rotation = 0,
    this.fog = !1,
    this.setValues(e)
}
,
n.SpriteMaterial.prototype = Object.create(n.Material.prototype),
n.SpriteMaterial.prototype.constructor = n.SpriteMaterial,
n.SpriteMaterial.prototype.copy = function(e) {
    return n.Material.prototype.copy.call(this, e),
    this.color.copy(e.color),
    this.map = e.map,
    this.rotation = e.rotation,
    this.fog = e.fog,
    this
}
,
n.Texture = function(e, t, i, r, o, a, s, l, h) {
    Object.defineProperty(this, "id", {
        value: n.TextureIdCount++
    }),
    this.uuid = n.Math.generateUUID(),
    this.name = "",
    this.sourceFile = "",
    this.image = void 0 !== e ? e : n.Texture.DEFAULT_IMAGE,
    this.mipmaps = [],
    this.mapping = void 0 !== t ? t : n.Texture.DEFAULT_MAPPING,
    this.wrapS = void 0 !== i ? i : n.ClampToEdgeWrapping,
    this.wrapT = void 0 !== r ? r : n.ClampToEdgeWrapping,
    this.magFilter = void 0 !== o ? o : n.LinearFilter,
    this.minFilter = void 0 !== a ? a : n.LinearMipMapLinearFilter,
    this.anisotropy = void 0 !== h ? h : 1,
    this.format = void 0 !== s ? s : n.RGBAFormat,
    this.type = void 0 !== l ? l : n.UnsignedByteType,
    this.offset = new n.Vector2(0,0),
    this.repeat = new n.Vector2(1,1),
    this.generateMipmaps = !0,
    this.premultiplyAlpha = !1,
    this.flipY = !0,
    this.unpackAlignment = 4,
    this.encoding = n.LinearEncoding,
    this.version = 0,
    this.onUpdate = null
}
,
n.Texture.DEFAULT_IMAGE = void 0,
n.Texture.DEFAULT_MAPPING = n.UVMapping,
n.Texture.prototype = {
    constructor: n.Texture,
    set needsUpdate(e) {
        e === !0 && this.version++
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.image = e.image,
        this.mipmaps = e.mipmaps.slice(0),
        this.mapping = e.mapping,
        this.wrapS = e.wrapS,
        this.wrapT = e.wrapT,
        this.magFilter = e.magFilter,
        this.minFilter = e.minFilter,
        this.anisotropy = e.anisotropy,
        this.format = e.format,
        this.type = e.type,
        this.offset.copy(e.offset),
        this.repeat.copy(e.repeat),
        this.generateMipmaps = e.generateMipmaps,
        this.premultiplyAlpha = e.premultiplyAlpha,
        this.flipY = e.flipY,
        this.unpackAlignment = e.unpackAlignment,
        this.encoding = e.encoding,
        this
    },
    toJSON: function(e) {
        function t(e) {
            var t;
            return void 0 !== e.toDataURL ? t = e : (t = document.createElement("canvas"),
            t.width = e.width,
            t.height = e.height,
            t.getContext("2d").drawImage(e, 0, 0, e.width, e.height)),
            t.width > 2048 || t.height > 2048 ? t.toDataURL("image/jpeg", .6) : t.toDataURL("image/png")
        }
        if (void 0 !== e.textures[this.uuid])
            return e.textures[this.uuid];
        var i = {
            metadata: {
                version: 4.4,
                type: "Texture",
                generator: "Texture.toJSON"
            },
            uuid: this.uuid,
            name: this.name,
            mapping: this.mapping,
            repeat: [this.repeat.x, this.repeat.y],
            offset: [this.offset.x, this.offset.y],
            wrap: [this.wrapS, this.wrapT],
            minFilter: this.minFilter,
            magFilter: this.magFilter,
            anisotropy: this.anisotropy
        };
        if (void 0 !== this.image) {
            var r = this.image;
            void 0 === r.uuid && (r.uuid = n.Math.generateUUID()),
            void 0 === e.images[r.uuid] && (e.images[r.uuid] = {
                uuid: r.uuid,
                url: t(r)
            }),
            i.image = r.uuid
        }
        return e.textures[this.uuid] = i,
        i
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    },
    transformUv: function(e) {
        if (this.mapping === n.UVMapping) {
            if (e.multiply(this.repeat),
            e.add(this.offset),
            e.x < 0 || e.x > 1)
                switch (this.wrapS) {
                case n.RepeatWrapping:
                    e.x = e.x - Math.floor(e.x);
                    break;
                case n.ClampToEdgeWrapping:
                    e.x = e.x < 0 ? 0 : 1;
                    break;
                case n.MirroredRepeatWrapping:
                    1 === Math.abs(Math.floor(e.x) % 2) ? e.x = Math.ceil(e.x) - e.x : e.x = e.x - Math.floor(e.x)
                }
            if (e.y < 0 || e.y > 1)
                switch (this.wrapT) {
                case n.RepeatWrapping:
                    e.y = e.y - Math.floor(e.y);
                    break;
                case n.ClampToEdgeWrapping:
                    e.y = e.y < 0 ? 0 : 1;
                    break;
                case n.MirroredRepeatWrapping:
                    1 === Math.abs(Math.floor(e.y) % 2) ? e.y = Math.ceil(e.y) - e.y : e.y = e.y - Math.floor(e.y)
                }
            this.flipY && (e.y = 1 - e.y)
        }
    }
},
n.EventDispatcher.prototype.apply(n.Texture.prototype),
n.TextureIdCount = 0,
n.CanvasTexture = function(e, t, i, r, o, a, s, l, h) {
    n.Texture.call(this, e, t, i, r, o, a, s, l, h),
    this.needsUpdate = !0
}
,
n.CanvasTexture.prototype = Object.create(n.Texture.prototype),
n.CanvasTexture.prototype.constructor = n.CanvasTexture,
n.CubeTexture = function(e, t, i, r, o, a, s, l, h) {
    e = void 0 !== e ? e : [],
    t = void 0 !== t ? t : n.CubeReflectionMapping,
    n.Texture.call(this, e, t, i, r, o, a, s, l, h),
    this.flipY = !1
}
,
n.CubeTexture.prototype = Object.create(n.Texture.prototype),
n.CubeTexture.prototype.constructor = n.CubeTexture,
Object.defineProperty(n.CubeTexture.prototype, "images", {
    get: function() {
        return this.image
    },
    set: function(e) {
        this.image = e
    }
}),
n.CompressedTexture = function(e, t, i, r, o, a, s, l, h, c, u) {
    n.Texture.call(this, null, a, s, l, h, c, r, o, u),
    this.image = {
        width: t,
        height: i
    },
    this.mipmaps = e,
    this.flipY = !1,
    this.generateMipmaps = !1
}
,
n.CompressedTexture.prototype = Object.create(n.Texture.prototype),
n.CompressedTexture.prototype.constructor = n.CompressedTexture,
n.DataTexture = function(e, t, i, r, o, a, s, l, h, c, u) {
    n.Texture.call(this, null, a, s, l, h, c, r, o, u),
    this.image = {
        data: e,
        width: t,
        height: i
    },
    this.magFilter = void 0 !== h ? h : n.NearestFilter,
    this.minFilter = void 0 !== c ? c : n.NearestFilter,
    this.flipY = !1,
    this.generateMipmaps = !1
}
,
n.DataTexture.prototype = Object.create(n.Texture.prototype),
n.DataTexture.prototype.constructor = n.DataTexture,
n.VideoTexture = function(e, t, i, r, o, a, s, l, h) {
    function c() {
        requestAnimationFrame(c),
        e.readyState === e.HAVE_ENOUGH_DATA && (u.needsUpdate = !0)
    }
    n.Texture.call(this, e, t, i, r, o, a, s, l, h),
    this.generateMipmaps = !1;
    var u = this;
    c()
}
,
n.VideoTexture.prototype = Object.create(n.Texture.prototype),
n.VideoTexture.prototype.constructor = n.VideoTexture,
n.Group = function() {
    n.Object3D.call(this),
    this.type = "Group"
}
,
n.Group.prototype = Object.create(n.Object3D.prototype),
n.Group.prototype.constructor = n.Group,
n.Points = function(e, t) {
    n.Object3D.call(this),
    this.type = "Points",
    this.geometry = void 0 !== e ? e : new n.Geometry,
    this.material = void 0 !== t ? t : new n.PointsMaterial({
        color: 16777215 * Math.random()
    })
}
,
n.Points.prototype = Object.create(n.Object3D.prototype),
n.Points.prototype.constructor = n.Points,
n.Points.prototype.raycast = function() {
    var e = new n.Matrix4
      , t = new n.Ray
      , i = new n.Sphere;
    return function(r, o) {
        function a(e, i) {
            var n = t.distanceSqToPoint(e);
            if (n < d) {
                var a = t.closestPointToPoint(e);
                a.applyMatrix4(h);
                var l = r.ray.origin.distanceTo(a);
                if (l < r.near || l > r.far)
                    return;
                o.push({
                    distance: l,
                    distanceToRay: Math.sqrt(n),
                    point: a.clone(),
                    index: i,
                    face: null,
                    object: s
                })
            }
        }
        var s = this
          , l = this.geometry
          , h = this.matrixWorld
          , c = r.params.Points.threshold;
        if (null === l.boundingSphere && l.computeBoundingSphere(),
        i.copy(l.boundingSphere),
        i.applyMatrix4(h),
        r.ray.intersectsSphere(i) !== !1) {
            e.getInverse(h),
            t.copy(r.ray).applyMatrix4(e);
            var u = c / ((this.scale.x + this.scale.y + this.scale.z) / 3)
              , d = u * u
              , p = new n.Vector3;
            if (l instanceof n.BufferGeometry) {
                var f = l.index
                  , g = l.attributes
                  , m = g.position.array;
                if (null !== f)
                    for (var v = f.array, A = 0, y = v.length; A < y; A++) {
                        var C = v[A];
                        p.fromArray(m, 3 * C),
                        a(p, C)
                    }
                else
                    for (var A = 0, I = m.length / 3; A < I; A++)
                        p.fromArray(m, 3 * A),
                        a(p, A)
            } else
                for (var b = l.vertices, A = 0, I = b.length; A < I; A++)
                    a(b[A], A)
        }
    }
}(),
n.Points.prototype.clone = function() {
    return new this.constructor(this.geometry,this.material).copy(this)
}
,
n.Line = function(e, t, i) {
    return 1 === i ? (console.warn("THREE.Line: parameter THREE.LinePieces no longer supported. Created THREE.LineSegments instead."),
    new n.LineSegments(e,t)) : (n.Object3D.call(this),
    this.type = "Line",
    this.geometry = void 0 !== e ? e : new n.Geometry,
    void (this.material = void 0 !== t ? t : new n.LineBasicMaterial({
        color: 16777215 * Math.random()
    })))
}
,
n.Line.prototype = Object.create(n.Object3D.prototype),
n.Line.prototype.constructor = n.Line,
n.Line.prototype.raycast = function() {
    var e = new n.Matrix4
      , t = new n.Ray
      , i = new n.Sphere;
    return function(r, o) {
        var a = r.linePrecision
          , s = a * a
          , l = this.geometry
          , h = this.matrixWorld;
        if (null === l.boundingSphere && l.computeBoundingSphere(),
        i.copy(l.boundingSphere),
        i.applyMatrix4(h),
        r.ray.intersectsSphere(i) !== !1) {
            e.getInverse(h),
            t.copy(r.ray).applyMatrix4(e);
            var c = new n.Vector3
              , u = new n.Vector3
              , d = new n.Vector3
              , p = new n.Vector3
              , f = this instanceof n.LineSegments ? 2 : 1;
            if (l instanceof n.BufferGeometry) {
                var g = l.index
                  , m = l.attributes
                  , v = m.position.array;
                if (null !== g)
                    for (var A = g.array, y = 0, C = A.length - 1; y < C; y += f) {
                        var I = A[y]
                          , b = A[y + 1];
                        c.fromArray(v, 3 * I),
                        u.fromArray(v, 3 * b);
                        var w = t.distanceSqToSegment(c, u, p, d);
                        if (!(w > s)) {
                            p.applyMatrix4(this.matrixWorld);
                            var E = r.ray.origin.distanceTo(p);
                            E < r.near || E > r.far || o.push({
                                distance: E,
                                point: d.clone().applyMatrix4(this.matrixWorld),
                                index: y,
                                face: null,
                                faceIndex: null,
                                object: this
                            })
                        }
                    }
                else
                    for (var y = 0, C = v.length / 3 - 1; y < C; y += f) {
                        c.fromArray(v, 3 * y),
                        u.fromArray(v, 3 * y + 3);
                        var w = t.distanceSqToSegment(c, u, p, d);
                        if (!(w > s)) {
                            p.applyMatrix4(this.matrixWorld);
                            var E = r.ray.origin.distanceTo(p);
                            E < r.near || E > r.far || o.push({
                                distance: E,
                                point: d.clone().applyMatrix4(this.matrixWorld),
                                index: y,
                                face: null,
                                faceIndex: null,
                                object: this
                            })
                        }
                    }
            } else if (l instanceof n.Geometry)
                for (var x = l.vertices, T = x.length, y = 0; y < T - 1; y += f) {
                    var w = t.distanceSqToSegment(x[y], x[y + 1], p, d);
                    if (!(w > s)) {
                        p.applyMatrix4(this.matrixWorld);
                        var E = r.ray.origin.distanceTo(p);
                        E < r.near || E > r.far || o.push({
                            distance: E,
                            point: d.clone().applyMatrix4(this.matrixWorld),
                            index: y,
                            face: null,
                            faceIndex: null,
                            object: this
                        })
                    }
                }
        }
    }
}(),
n.Line.prototype.clone = function() {
    return new this.constructor(this.geometry,this.material).copy(this)
}
,
n.LineStrip = 0,
n.LinePieces = 1,
n.LineSegments = function(e, t) {
    n.Line.call(this, e, t),
    this.type = "LineSegments"
}
,
n.LineSegments.prototype = Object.create(n.Line.prototype),
n.LineSegments.prototype.constructor = n.LineSegments,
n.Mesh = function(e, t) {
    n.Object3D.call(this),
    this.type = "Mesh",
    this.geometry = void 0 !== e ? e : new n.Geometry,
    this.material = void 0 !== t ? t : new n.MeshBasicMaterial({
        color: 16777215 * Math.random()
    }),
    this.drawMode = n.TrianglesDrawMode,
    this.updateMorphTargets()
}
,
n.Mesh.prototype = Object.create(n.Object3D.prototype),
n.Mesh.prototype.constructor = n.Mesh,
n.Mesh.prototype.setDrawMode = function(e) {
    this.drawMode = e
}
,
n.Mesh.prototype.updateMorphTargets = function() {
    if (void 0 !== this.geometry.morphTargets && this.geometry.morphTargets.length > 0) {
        this.morphTargetBase = -1,
        this.morphTargetInfluences = [],
        this.morphTargetDictionary = {};
        for (var e = 0, t = this.geometry.morphTargets.length; e < t; e++)
            this.morphTargetInfluences.push(0),
            this.morphTargetDictionary[this.geometry.morphTargets[e].name] = e
    }
}
,
n.Mesh.prototype.getMorphTargetIndexByName = function(e) {
    return void 0 !== this.morphTargetDictionary[e] ? this.morphTargetDictionary[e] : (console.warn("THREE.Mesh.getMorphTargetIndexByName: morph target " + e + " does not exist. Returning 0."),
    0)
}
,
n.Mesh.prototype.raycast = function() {
    function e(e, t, i, r, o, a, s) {
        return n.Triangle.barycoordFromPoint(e, t, i, r, m),
        o.multiplyScalar(m.x),
        a.multiplyScalar(m.y),
        s.multiplyScalar(m.z),
        o.add(a).add(s),
        o.clone()
    }
    function t(e, t, i, r, o, a, s) {
        var l, h = e.material;
        if (l = h.side === n.BackSide ? i.intersectTriangle(a, o, r, !0, s) : i.intersectTriangle(r, o, a, h.side !== n.DoubleSide, s),
        null === l)
            return null;
        A.copy(s),
        A.applyMatrix4(e.matrixWorld);
        var c = t.ray.origin.distanceTo(A);
        return c < t.near || c > t.far ? null : {
            distance: c,
            point: A.clone(),
            object: e
        }
    }
    function i(i, r, o, a, c, u, d, m) {
        s.fromArray(a, 3 * u),
        l.fromArray(a, 3 * d),
        h.fromArray(a, 3 * m);
        var A = t(i, r, o, s, l, h, v);
        return A && (c && (p.fromArray(c, 2 * u),
        f.fromArray(c, 2 * d),
        g.fromArray(c, 2 * m),
        A.uv = e(v, s, l, h, p, f, g)),
        A.face = new n.Face3(u,d,m,n.Triangle.normal(s, l, h)),
        A.faceIndex = u),
        A
    }
    var r = new n.Matrix4
      , o = new n.Ray
      , a = new n.Sphere
      , s = new n.Vector3
      , l = new n.Vector3
      , h = new n.Vector3
      , c = new n.Vector3
      , u = new n.Vector3
      , d = new n.Vector3
      , p = new n.Vector2
      , f = new n.Vector2
      , g = new n.Vector2
      , m = new n.Vector3
      , v = new n.Vector3
      , A = new n.Vector3;
    return function(m, A) {
        var y = this.geometry
          , C = this.material
          , I = this.matrixWorld;
        if (void 0 !== C && (null === y.boundingSphere && y.computeBoundingSphere(),
        a.copy(y.boundingSphere),
        a.applyMatrix4(I),
        m.ray.intersectsSphere(a) !== !1 && (r.getInverse(I),
        o.copy(m.ray).applyMatrix4(r),
        null === y.boundingBox || o.intersectsBox(y.boundingBox) !== !1))) {
            var b, w;
            if (y instanceof n.BufferGeometry) {
                var E, x, T, M = y.index, S = y.attributes, _ = S.position.array;
                if (void 0 !== S.uv && (b = S.uv.array),
                null !== M)
                    for (var P = M.array, R = 0, L = P.length; R < L; R += 3)
                        E = P[R],
                        x = P[R + 1],
                        T = P[R + 2],
                        w = i(this, m, o, _, b, E, x, T),
                        w && (w.faceIndex = Math.floor(R / 3),
                        A.push(w));
                else
                    for (var R = 0, L = _.length; R < L; R += 9)
                        E = R / 3,
                        x = E + 1,
                        T = E + 2,
                        w = i(this, m, o, _, b, E, x, T),
                        w && (w.index = E,
                        A.push(w))
            } else if (y instanceof n.Geometry) {
                var O, D, F, N = C instanceof n.MultiMaterial, B = N === !0 ? C.materials : null, k = y.vertices, U = y.faces, V = y.faceVertexUvs[0];
                V.length > 0 && (b = V);
                for (var z = 0, G = U.length; z < G; z++) {
                    var H = U[z]
                      , W = N === !0 ? B[H.materialIndex] : C;
                    if (void 0 !== W) {
                        if (O = k[H.a],
                        D = k[H.b],
                        F = k[H.c],
                        W.morphTargets === !0) {
                            var j = y.morphTargets
                              , Y = this.morphTargetInfluences;
                            s.set(0, 0, 0),
                            l.set(0, 0, 0),
                            h.set(0, 0, 0);
                            for (var X = 0, Z = j.length; X < Z; X++) {
                                var Q = Y[X];
                                if (0 !== Q) {
                                    var q = j[X].vertices;
                                    s.addScaledVector(c.subVectors(q[H.a], O), Q),
                                    l.addScaledVector(u.subVectors(q[H.b], D), Q),
                                    h.addScaledVector(d.subVectors(q[H.c], F), Q)
                                }
                            }
                            s.add(O),
                            l.add(D),
                            h.add(F),
                            O = s,
                            D = l,
                            F = h
                        }
                        if (w = t(this, m, o, O, D, F, v)) {
                            if (b) {
                                var K = b[z];
                                p.copy(K[0]),
                                f.copy(K[1]),
                                g.copy(K[2]),
                                w.uv = e(v, O, D, F, p, f, g)
                            }
                            w.face = H,
                            w.faceIndex = z,
                            A.push(w)
                        }
                    }
                }
            }
        }
    }
}(),
n.Mesh.prototype.clone = function() {
    return new this.constructor(this.geometry,this.material).copy(this)
}
,
n.Bone = function(e) {
    n.Object3D.call(this),
    this.type = "Bone",
    this.skin = e
}
,
n.Bone.prototype = Object.create(n.Object3D.prototype),
n.Bone.prototype.constructor = n.Bone,
n.Bone.prototype.copy = function(e) {
    return n.Object3D.prototype.copy.call(this, e),
    this.skin = e.skin,
    this
}
,
n.Skeleton = function(e, t, i) {
    if (this.useVertexTexture = void 0 === i || i,
    this.identityMatrix = new n.Matrix4,
    e = e || [],
    this.bones = e.slice(0),
    this.useVertexTexture) {
        var r = Math.sqrt(4 * this.bones.length);
        r = n.Math.nextPowerOfTwo(Math.ceil(r)),
        r = Math.max(r, 4),
        this.boneTextureWidth = r,
        this.boneTextureHeight = r,
        this.boneMatrices = new Float32Array(this.boneTextureWidth * this.boneTextureHeight * 4),
        this.boneTexture = new n.DataTexture(this.boneMatrices,this.boneTextureWidth,this.boneTextureHeight,n.RGBAFormat,n.FloatType)
    } else
        this.boneMatrices = new Float32Array(16 * this.bones.length);
    if (void 0 === t)
        this.calculateInverses();
    else if (this.bones.length === t.length)
        this.boneInverses = t.slice(0);
    else {
        console.warn("THREE.Skeleton bonInverses is the wrong length."),
        this.boneInverses = [];
        for (var o = 0, a = this.bones.length; o < a; o++)
            this.boneInverses.push(new n.Matrix4)
    }
}
,
n.Skeleton.prototype.calculateInverses = function() {
    this.boneInverses = [];
    for (var e = 0, t = this.bones.length; e < t; e++) {
        var i = new n.Matrix4;
        this.bones[e] && i.getInverse(this.bones[e].matrixWorld),
        this.boneInverses.push(i)
    }
}
,
n.Skeleton.prototype.pose = function() {
    for (var e, t = 0, i = this.bones.length; t < i; t++)
        e = this.bones[t],
        e && e.matrixWorld.getInverse(this.boneInverses[t]);
    for (var t = 0, i = this.bones.length; t < i; t++)
        e = this.bones[t],
        e && (e.parent ? (e.matrix.getInverse(e.parent.matrixWorld),
        e.matrix.multiply(e.matrixWorld)) : e.matrix.copy(e.matrixWorld),
        e.matrix.decompose(e.position, e.quaternion, e.scale))
}
,
n.Skeleton.prototype.update = function() {
    var e = new n.Matrix4;
    return function() {
        for (var t = 0, i = this.bones.length; t < i; t++) {
            var n = this.bones[t] ? this.bones[t].matrixWorld : this.identityMatrix;
            e.multiplyMatrices(n, this.boneInverses[t]),
            e.flattenToArrayOffset(this.boneMatrices, 16 * t)
        }
        this.useVertexTexture && (this.boneTexture.needsUpdate = !0)
    }
}(),
n.Skeleton.prototype.clone = function() {
    return new n.Skeleton(this.bones,this.boneInverses,this.useVertexTexture)
}
,
n.SkinnedMesh = function(e, t, i) {
    n.Mesh.call(this, e, t),
    this.type = "SkinnedMesh",
    this.bindMode = "attached",
    this.bindMatrix = new n.Matrix4,
    this.bindMatrixInverse = new n.Matrix4;
    var r = [];
    if (this.geometry && void 0 !== this.geometry.bones) {
        for (var o, a, s = 0, l = this.geometry.bones.length; s < l; ++s)
            a = this.geometry.bones[s],
            o = new n.Bone(this),
            r.push(o),
            o.name = a.name,
            o.position.fromArray(a.pos),
            o.quaternion.fromArray(a.rotq),
            void 0 !== a.scl && o.scale.fromArray(a.scl);
        for (var s = 0, l = this.geometry.bones.length; s < l; ++s)
            a = this.geometry.bones[s],
            a.parent !== -1 && null !== a.parent ? r[a.parent].add(r[s]) : this.add(r[s])
    }
    this.normalizeSkinWeights(),
    this.updateMatrixWorld(!0),
    this.bind(new n.Skeleton(r,void 0,i), this.matrixWorld)
}
,
n.SkinnedMesh.prototype = Object.create(n.Mesh.prototype),
n.SkinnedMesh.prototype.constructor = n.SkinnedMesh,
n.SkinnedMesh.prototype.bind = function(e, t) {
    this.skeleton = e,
    void 0 === t && (this.updateMatrixWorld(!0),
    this.skeleton.calculateInverses(),
    t = this.matrixWorld),
    this.bindMatrix.copy(t),
    this.bindMatrixInverse.getInverse(t)
}
,
n.SkinnedMesh.prototype.pose = function() {
    this.skeleton.pose()
}
,
n.SkinnedMesh.prototype.normalizeSkinWeights = function() {
    if (this.geometry instanceof n.Geometry)
        for (var e = 0; e < this.geometry.skinWeights.length; e++) {
            var t = this.geometry.skinWeights[e]
              , i = 1 / t.lengthManhattan();
            i !== 1 / 0 ? t.multiplyScalar(i) : t.set(1, 0, 0, 0)
        }
    else if (this.geometry instanceof n.BufferGeometry)
        for (var r = new n.Vector4, o = this.geometry.attributes.skinWeight, e = 0; e < o.count; e++) {
            r.x = o.getX(e),
            r.y = o.getY(e),
            r.z = o.getZ(e),
            r.w = o.getW(e);
            var i = 1 / r.lengthManhattan();
            i !== 1 / 0 ? r.multiplyScalar(i) : r.set(1, 0, 0, 0),
            o.setXYZW(e, r.x, r.y, r.z, r.w)
        }
}
,
n.SkinnedMesh.prototype.updateMatrixWorld = function(e) {
    n.Mesh.prototype.updateMatrixWorld.call(this, !0),
    "attached" === this.bindMode ? this.bindMatrixInverse.getInverse(this.matrixWorld) : "detached" === this.bindMode ? this.bindMatrixInverse.getInverse(this.bindMatrix) : console.warn("THREE.SkinnedMesh unrecognized bindMode: " + this.bindMode)
}
,
n.SkinnedMesh.prototype.clone = function() {
    return new this.constructor(this.geometry,this.material,this.useVertexTexture).copy(this)
}
,
n.LOD = function() {
    n.Object3D.call(this),
    this.type = "LOD",
    Object.defineProperties(this, {
        levels: {
            enumerable: !0,
            value: []
        },
        objects: {
            get: function() {
                return console.warn("THREE.LOD: .objects has been renamed to .levels."),
                this.levels
            }
        }
    })
}
,
n.LOD.prototype = Object.create(n.Object3D.prototype),
n.LOD.prototype.constructor = n.LOD,
n.LOD.prototype.addLevel = function(e, t) {
    void 0 === t && (t = 0),
    t = Math.abs(t);
    for (var i = this.levels, n = 0; n < i.length && !(t < i[n].distance); n++)
        ;
    i.splice(n, 0, {
        distance: t,
        object: e
    }),
    this.add(e)
}
,
n.LOD.prototype.getObjectForDistance = function(e) {
    for (var t = this.levels, i = 1, n = t.length; i < n && !(e < t[i].distance); i++)
        ;
    return t[i - 1].object
}
,
n.LOD.prototype.raycast = function() {
    var e = new n.Vector3;
    return function(t, i) {
        e.setFromMatrixPosition(this.matrixWorld);
        var n = t.ray.origin.distanceTo(e);
        this.getObjectForDistance(n).raycast(t, i)
    }
}(),
n.LOD.prototype.update = function() {
    var e = new n.Vector3
      , t = new n.Vector3;
    return function(i) {
        var n = this.levels;
        if (n.length > 1) {
            e.setFromMatrixPosition(i.matrixWorld),
            t.setFromMatrixPosition(this.matrixWorld);
            var r = e.distanceTo(t);
            n[0].object.visible = !0;
            for (var o = 1, a = n.length; o < a && r >= n[o].distance; o++)
                n[o - 1].object.visible = !1,
                n[o].object.visible = !0;
            for (; o < a; o++)
                n[o].object.visible = !1
        }
    }
}(),
n.LOD.prototype.copy = function(e) {
    n.Object3D.prototype.copy.call(this, e, !1);
    for (var t = e.levels, i = 0, r = t.length; i < r; i++) {
        var o = t[i];
        this.addLevel(o.object.clone(), o.distance)
    }
    return this
}
,
n.LOD.prototype.toJSON = function(e) {
    var t = n.Object3D.prototype.toJSON.call(this, e);
    t.object.levels = [];
    for (var i = this.levels, r = 0, o = i.length; r < o; r++) {
        var a = i[r];
        t.object.levels.push({
            object: a.object.uuid,
            distance: a.distance
        })
    }
    return t
}
,
n.Sprite = function() {
    var e = new Uint16Array([0, 1, 2, 0, 2, 3])
      , t = new Float32Array([-.5, -.5, 0, .5, -.5, 0, .5, .5, 0, -.5, .5, 0])
      , i = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1])
      , r = new n.BufferGeometry;
    return r.setIndex(new n.BufferAttribute(e,1)),
    r.addAttribute("position", new n.BufferAttribute(t,3)),
    r.addAttribute("uv", new n.BufferAttribute(i,2)),
    function(e) {
        n.Object3D.call(this),
        this.type = "Sprite",
        this.geometry = r,
        this.material = void 0 !== e ? e : new n.SpriteMaterial
    }
}(),
n.Sprite.prototype = Object.create(n.Object3D.prototype),
n.Sprite.prototype.constructor = n.Sprite,
n.Sprite.prototype.raycast = function() {
    var e = new n.Vector3;
    return function(t, i) {
        e.setFromMatrixPosition(this.matrixWorld);
        var n = t.ray.distanceSqToPoint(e)
          , r = this.scale.x * this.scale.y;
        n > r || i.push({
            distance: Math.sqrt(n),
            point: this.position,
            face: null,
            object: this
        })
    }
}(),
n.Sprite.prototype.clone = function() {
    return new this.constructor(this.material).copy(this)
}
,
n.Particle = n.Sprite,
n.LensFlare = function(e, t, i, r, o) {
    n.Object3D.call(this),
    this.lensFlares = [],
    this.positionScreen = new n.Vector3,
    this.customUpdateCallback = void 0,
    void 0 !== e && this.add(e, t, i, r, o)
}
,
n.LensFlare.prototype = Object.create(n.Object3D.prototype),
n.LensFlare.prototype.constructor = n.LensFlare,
n.LensFlare.prototype.add = function(e, t, i, r, o, a) {
    void 0 === t && (t = -1),
    void 0 === i && (i = 0),
    void 0 === a && (a = 1),
    void 0 === o && (o = new n.Color(16777215)),
    void 0 === r && (r = n.NormalBlending),
    i = Math.min(i, Math.max(0, i)),
    this.lensFlares.push({
        texture: e,
        size: t,
        distance: i,
        x: 0,
        y: 0,
        z: 0,
        scale: 1,
        rotation: 0,
        opacity: a,
        color: o,
        blending: r
    })
}
,
n.LensFlare.prototype.updateLensFlares = function() {
    var e, t, i = this.lensFlares.length, n = 2 * -this.positionScreen.x, r = 2 * -this.positionScreen.y;
    for (e = 0; e < i; e++)
        t = this.lensFlares[e],
        t.x = this.positionScreen.x + n * t.distance,
        t.y = this.positionScreen.y + r * t.distance,
        t.wantedRotation = t.x * Math.PI * .25,
        t.rotation += .25 * (t.wantedRotation - t.rotation)
}
,
n.LensFlare.prototype.copy = function(e) {
    n.Object3D.prototype.copy.call(this, e),
    this.positionScreen.copy(e.positionScreen),
    this.customUpdateCallback = e.customUpdateCallback;
    for (var t = 0, i = e.lensFlares.length; t < i; t++)
        this.lensFlares.push(e.lensFlares[t]);
    return this
}
,
n.Scene = function() {
    n.Object3D.call(this),
    this.type = "Scene",
    this.fog = null,
    this.overrideMaterial = null,
    this.autoUpdate = !0
}
,
n.Scene.prototype = Object.create(n.Object3D.prototype),
n.Scene.prototype.constructor = n.Scene,
n.Scene.prototype.copy = function(e, t) {
    return n.Object3D.prototype.copy.call(this, e, t),
    null !== e.fog && (this.fog = e.fog.clone()),
    null !== e.overrideMaterial && (this.overrideMaterial = e.overrideMaterial.clone()),
    this.autoUpdate = e.autoUpdate,
    this.matrixAutoUpdate = e.matrixAutoUpdate,
    this
}
,
n.Fog = function(e, t, i) {
    this.name = "",
    this.color = new n.Color(e),
    this.near = void 0 !== t ? t : 1,
    this.far = void 0 !== i ? i : 1e3
}
,
n.Fog.prototype.clone = function() {
    return new n.Fog(this.color.getHex(),this.near,this.far)
}
,
n.FogExp2 = function(e, t) {
    this.name = "",
    this.color = new n.Color(e),
    this.density = void 0 !== t ? t : 25e-5
}
,
n.FogExp2.prototype.clone = function() {
    return new n.FogExp2(this.color.getHex(),this.density)
}
,
n.ShaderChunk = {},
n.ShaderChunk.alphamap_fragment = "#ifdef USE_ALPHAMAP\n\tdiffuseColor.a *= texture2D( alphaMap, vUv ).g;\n#endif\n",
n.ShaderChunk.alphamap_pars_fragment = "#ifdef USE_ALPHAMAP\n\tuniform sampler2D alphaMap;\n#endif\n",
n.ShaderChunk.alphatest_fragment = "#ifdef ALPHATEST\n\tif ( diffuseColor.a < ALPHATEST ) discard;\n#endif\n",
n.ShaderChunk.aomap_fragment = "#ifdef USE_AOMAP\n\tfloat ambientOcclusion = ( texture2D( aoMap, vUv2 ).r - 1.0 ) * aoMapIntensity + 1.0;\n\treflectedLight.indirectDiffuse *= ambientOcclusion;\n\t#if defined( USE_ENVMAP ) && defined( STANDARD )\n\t\tfloat dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );\n\t\treflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.specularRoughness );\n\t#endif\n#endif\n",
n.ShaderChunk.aomap_pars_fragment = "#ifdef USE_AOMAP\n\tuniform sampler2D aoMap;\n\tuniform float aoMapIntensity;\n#endif",
n.ShaderChunk.begin_vertex = "\nvec3 transformed = vec3( position );\n",
n.ShaderChunk.beginnormal_vertex = "\nvec3 objectNormal = vec3( normal );\n",
n.ShaderChunk.bsdfs = "bool testLightInRange( const in float lightDistance, const in float cutoffDistance ) {\n\treturn any( bvec2( cutoffDistance == 0.0, lightDistance < cutoffDistance ) );\n}\nfloat punctualLightIntensityToIrradianceFactor( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {\n\t\tif( decayExponent > 0.0 ) {\n#if defined ( PHYSICALLY_CORRECT_LIGHTS )\n\t\t\tfloat distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );\n\t\t\tfloat maxDistanceCutoffFactor = pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );\n\t\t\treturn distanceFalloff * maxDistanceCutoffFactor;\n#else\n\t\t\treturn pow( saturate( -lightDistance / cutoffDistance + 1.0 ), decayExponent );\n#endif\n\t\t}\n\t\treturn 1.0;\n}\nvec3 BRDF_Diffuse_Lambert( const in vec3 diffuseColor ) {\n\treturn RECIPROCAL_PI * diffuseColor;\n}\nvec3 F_Schlick( const in vec3 specularColor, const in float dotLH ) {\n\tfloat fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );\n\treturn ( 1.0 - specularColor ) * fresnel + specularColor;\n}\nfloat G_GGX_Smith( const in float alpha, const in float dotNL, const in float dotNV ) {\n\tfloat a2 = pow2( alpha );\n\tfloat gl = dotNL + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );\n\tfloat gv = dotNV + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );\n\treturn 1.0 / ( gl * gv );\n}\nfloat D_GGX( const in float alpha, const in float dotNH ) {\n\tfloat a2 = pow2( alpha );\n\tfloat denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;\n\treturn RECIPROCAL_PI * a2 / pow2( denom );\n}\nvec3 BRDF_Specular_GGX( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {\n\tfloat alpha = pow2( roughness );\n\tvec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );\n\tfloat dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );\n\tfloat dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );\n\tfloat dotNH = saturate( dot( geometry.normal, halfDir ) );\n\tfloat dotLH = saturate( dot( incidentLight.direction, halfDir ) );\n\tvec3 F = F_Schlick( specularColor, dotLH );\n\tfloat G = G_GGX_Smith( alpha, dotNL, dotNV );\n\tfloat D = D_GGX( alpha, dotNH );\n\treturn F * ( G * D );\n}\nvec3 BRDF_Specular_GGX_Environment( const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {\n\tfloat dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );\n\tconst vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );\n\tconst vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );\n\tvec4 r = roughness * c0 + c1;\n\tfloat a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;\n\tvec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;\n\treturn specularColor * AB.x + AB.y;\n}\nfloat G_BlinnPhong_Implicit( ) {\n\treturn 0.25;\n}\nfloat D_BlinnPhong( const in float shininess, const in float dotNH ) {\n\treturn RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );\n}\nvec3 BRDF_Specular_BlinnPhong( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float shininess ) {\n\tvec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );\n\tfloat dotNH = saturate( dot( geometry.normal, halfDir ) );\n\tfloat dotLH = saturate( dot( incidentLight.direction, halfDir ) );\n\tvec3 F = F_Schlick( specularColor, dotLH );\n\tfloat G = G_BlinnPhong_Implicit( );\n\tfloat D = D_BlinnPhong( shininess, dotNH );\n\treturn F * ( G * D );\n}\nfloat GGXRoughnessToBlinnExponent( const in float ggxRoughness ) {\n\treturn ( 2.0 / pow2( ggxRoughness + 0.0001 ) - 2.0 );\n}\nfloat BlinnExponentToGGXRoughness( const in float blinnExponent ) {\n\treturn sqrt( 2.0 / ( blinnExponent + 2.0 ) );\n}\n",
n.ShaderChunk.bumpmap_pars_fragment = "#ifdef USE_BUMPMAP\n\tuniform sampler2D bumpMap;\n\tuniform float bumpScale;\n\tvec2 dHdxy_fwd() {\n\t\tvec2 dSTdx = dFdx( vUv );\n\t\tvec2 dSTdy = dFdy( vUv );\n\t\tfloat Hll = bumpScale * texture2D( bumpMap, vUv ).x;\n\t\tfloat dBx = bumpScale * texture2D( bumpMap, vUv + dSTdx ).x - Hll;\n\t\tfloat dBy = bumpScale * texture2D( bumpMap, vUv + dSTdy ).x - Hll;\n\t\treturn vec2( dBx, dBy );\n\t}\n\tvec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy ) {\n\t\tvec3 vSigmaX = dFdx( surf_pos );\n\t\tvec3 vSigmaY = dFdy( surf_pos );\n\t\tvec3 vN = surf_norm;\n\t\tvec3 R1 = cross( vSigmaY, vN );\n\t\tvec3 R2 = cross( vN, vSigmaX );\n\t\tfloat fDet = dot( vSigmaX, R1 );\n\t\tvec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );\n\t\treturn normalize( abs( fDet ) * surf_norm - vGrad );\n\t}\n#endif\n",
n.ShaderChunk.color_fragment = "#ifdef USE_COLOR\n\tdiffuseColor.rgb *= vColor;\n#endif",
n.ShaderChunk.color_pars_fragment = "#ifdef USE_COLOR\n\tvarying vec3 vColor;\n#endif\n",
n.ShaderChunk.color_pars_vertex = "#ifdef USE_COLOR\n\tvarying vec3 vColor;\n#endif",
n.ShaderChunk.color_vertex = "#ifdef USE_COLOR\n\tvColor.xyz = color.xyz;\n#endif",
n.ShaderChunk.common = "#define PI 3.14159\n#define PI2 6.28318\n#define RECIPROCAL_PI 0.31830988618\n#define RECIPROCAL_PI2 0.15915494\n#define LOG2 1.442695\n#define EPSILON 1e-6\n#define saturate(a) clamp( a, 0.0, 1.0 )\n#define whiteCompliment(a) ( 1.0 - saturate( a ) )\nfloat pow2( const in float x ) { return x*x; }\nfloat pow3( const in float x ) { return x*x*x; }\nfloat pow4( const in float x ) { float x2 = x*x; return x2*x2; }\nfloat average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }\nstruct IncidentLight {\n\tvec3 color;\n\tvec3 direction;\n\tbool visible;\n};\nstruct ReflectedLight {\n\tvec3 directDiffuse;\n\tvec3 directSpecular;\n\tvec3 indirectDiffuse;\n\tvec3 indirectSpecular;\n};\nstruct GeometricContext {\n\tvec3 position;\n\tvec3 normal;\n\tvec3 viewDir;\n};\nvec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n\treturn normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n}\nvec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {\n\treturn normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );\n}\nvec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {\n\tfloat distance = dot( planeNormal, point - pointOnPlane );\n\treturn - distance * planeNormal + point;\n}\nfloat sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {\n\treturn sign( dot( point - pointOnPlane, planeNormal ) );\n}\nvec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {\n\treturn lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;\n}\n",
n.ShaderChunk.cube_uv_reflection_fragment = "#ifdef ENVMAP_TYPE_CUBE_UV\nconst float cubeUV_textureSize = 1024.0;\nint getFaceFromDirection(vec3 direction) {\n    vec3 absDirection = abs(direction);\n    int face = -1;\n    if( absDirection.x > absDirection.z ) {\n        if(absDirection.x > absDirection.y )\n            face = direction.x > 0.0 ? 0 : 3;\n        else\n            face = direction.y > 0.0 ? 1 : 4;\n    }\n    else {\n        if(absDirection.z > absDirection.y )\n            face = direction.z > 0.0 ? 2 : 5;\n        else\n            face = direction.y > 0.0 ? 1 : 4;\n    }\n    return face;\n}\nconst float cubeUV_maxLods1 = log2(cubeUV_textureSize*0.25) - 1.0;\nconst float cubeUV_rangeClamp = exp2((6.0 - 1.0) * 2.0);\nvec2 MipLevelInfo( vec3 vec, float roughnessLevel, float roughness ) {\n    float scale = exp2(cubeUV_maxLods1 - roughnessLevel);\n    float dxRoughness = dFdx(roughness);\n    float dyRoughness = dFdy(roughness);\n    vec3 dx = dFdx( vec * scale * dxRoughness );\n    vec3 dy = dFdy( vec * scale * dyRoughness );\n    float d = max( dot( dx, dx ), dot( dy, dy ) );\n    d = clamp(d, 1.0, cubeUV_rangeClamp);\n    float mipLevel = 0.5 * log2(d);\n    return vec2(floor(mipLevel), fract(mipLevel));\n}\nconst float cubeUV_maxLods2 = log2(cubeUV_textureSize*0.25) - 2.0;\nconst float cubeUV_rcpTextureSize = 1.0 / cubeUV_textureSize;\nvec2 getCubeUV(vec3 direction, float roughnessLevel, float mipLevel) {\n    mipLevel = roughnessLevel > cubeUV_maxLods2 - 3.0 ? 0.0 : mipLevel;\n    float a = 16.0 * cubeUV_rcpTextureSize;\n    vec2 exp2_packed = exp2( vec2( roughnessLevel, mipLevel ) );\n    vec2 rcp_exp2_packed = vec2( 1.0 ) / exp2_packed;\n    float powScale = exp2_packed.x * exp2_packed.y;\n    float scale = rcp_exp2_packed.x * rcp_exp2_packed.y * 0.25;\n    float mipOffset = 0.75*(1.0 - rcp_exp2_packed.y) * rcp_exp2_packed.x;\n    bool bRes = mipLevel == 0.0;\n    scale =  bRes && (scale < a) ? a : scale;\n    vec3 r;\n    vec2 offset;\n    int face = getFaceFromDirection(direction);\n    float rcpPowScale = 1.0 / powScale;\n    if( face == 0) {\n        r = vec3(direction.x, -direction.z, direction.y);\n        offset = vec2(0.0+mipOffset,0.75 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  a : offset.y;\n    }\n    else if( face == 1) {\n        r = vec3(direction.y, direction.x, direction.z);\n        offset = vec2(scale+mipOffset, 0.75 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  a : offset.y;\n    }\n    else if( face == 2) {\n        r = vec3(direction.z, direction.x, direction.y);\n        offset = vec2(2.0*scale+mipOffset, 0.75 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  a : offset.y;\n    }\n    else if( face == 3) {\n        r = vec3(direction.x, direction.z, direction.y);\n        offset = vec2(0.0+mipOffset,0.5 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  0.0 : offset.y;\n    }\n    else if( face == 4) {\n        r = vec3(direction.y, direction.x, -direction.z);\n        offset = vec2(scale+mipOffset, 0.5 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  0.0 : offset.y;\n    }\n    else {\n        r = vec3(direction.z, -direction.x, direction.y);\n        offset = vec2(2.0*scale+mipOffset, 0.5 * rcpPowScale);\n        offset.y = bRes && (offset.y < 2.0*a) ?  0.0 : offset.y;\n    }\n    r = normalize(r);\n    float texelOffset = 0.5 * cubeUV_rcpTextureSize;\n    vec2 s = ( r.yz / abs( r.x ) + vec2( 1.0 ) ) * 0.5;\n    vec2 base = offset + vec2( texelOffset );\n    return base + s * ( scale - 2.0 * texelOffset );\n}\nconst float cubeUV_maxLods3 = log2(cubeUV_textureSize*0.25) - 3.0;\nvec4 textureCubeUV(vec3 reflectedDirection, float roughness ) {\n    float roughnessVal = roughness* cubeUV_maxLods3;\n    float r1 = floor(roughnessVal);\n    float r2 = r1 + 1.0;\n    float t = fract(roughnessVal);\n    vec2 mipInfo = MipLevelInfo(reflectedDirection, r1, roughness);\n    float s = mipInfo.y;\n    float level0 = mipInfo.x;\n    float level1 = level0 + 1.0;\n    level1 = level1 > 5.0 ? 5.0 : level1;\n    level0 += min( floor( s + 0.5 ), 5.0 );\n    vec2 uv_10 = getCubeUV(reflectedDirection, r1, level0);\n    vec4 color10 = envMapTexelToLinear(texture2D(envMap, uv_10));\n    vec2 uv_20 = getCubeUV(reflectedDirection, r2, level0);\n    vec4 color20 = envMapTexelToLinear(texture2D(envMap, uv_20));\n    vec4 result = mix(color10, color20, t);\n    return vec4(result.rgb, 1.0);\n}\n#endif\n",
n.ShaderChunk.defaultnormal_vertex = "#ifdef FLIP_SIDED\n\tobjectNormal = -objectNormal;\n#endif\nvec3 transformedNormal = normalMatrix * objectNormal;\n",
n.ShaderChunk.displacementmap_vertex = "#ifdef USE_DISPLACEMENTMAP\n\ttransformed += normal * ( texture2D( displacementMap, uv ).x * displacementScale + displacementBias );\n#endif\n",
n.ShaderChunk.displacementmap_pars_vertex = "#ifdef USE_DISPLACEMENTMAP\n\tuniform sampler2D displacementMap;\n\tuniform float displacementScale;\n\tuniform float displacementBias;\n#endif\n",
n.ShaderChunk.emissivemap_fragment = "#ifdef USE_EMISSIVEMAP\n\tvec4 emissiveColor = texture2D( emissiveMap, vUv );\n\temissiveColor.rgb = emissiveMapTexelToLinear( emissiveColor ).rgb;\n\ttotalEmissiveRadiance *= emissiveColor.rgb;\n#endif\n",
n.ShaderChunk.emissivemap_pars_fragment = "#ifdef USE_EMISSIVEMAP\n\tuniform sampler2D emissiveMap;\n#endif\n",
n.ShaderChunk.encodings_pars_fragment = "\nvec4 LinearToLinear( in vec4 value ) {\n  return value;\n}\nvec4 GammaToLinear( in vec4 value, in float gammaFactor ) {\n  return vec4( pow( value.xyz, vec3( gammaFactor ) ), value.w );\n}\nvec4 LinearToGamma( in vec4 value, in float gammaFactor ) {\n  return vec4( pow( value.xyz, vec3( 1.0 / gammaFactor ) ), value.w );\n}\nvec4 sRGBToLinear( in vec4 value ) {\n  return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.w );\n}\nvec4 LinearTosRGB( in vec4 value ) {\n  return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.w );\n}\nvec4 RGBEToLinear( in vec4 value ) {\n  return vec4( value.rgb * exp2( value.a * 255.0 - 128.0 ), 1.0 );\n}\nvec4 LinearToRGBE( in vec4 value ) {\n  float maxComponent = max( max( value.r, value.g ), value.b );\n  float fExp = clamp( ceil( log2( maxComponent ) ), -128.0, 127.0 );\n  return vec4( value.rgb / exp2( fExp ), ( fExp + 128.0 ) / 255.0 );\n}\nvec4 RGBMToLinear( in vec4 value, in float maxRange ) {\n  return vec4( value.xyz * value.w * maxRange, 1.0 );\n}\nvec4 LinearToRGBM( in vec4 value, in float maxRange ) {\n  float maxRGB = max( value.x, max( value.g, value.b ) );\n  float M      = clamp( maxRGB / maxRange, 0.0, 1.0 );\n  M            = ceil( M * 255.0 ) / 255.0;\n  return vec4( value.rgb / ( M * maxRange ), M );\n}\nvec4 RGBDToLinear( in vec4 value, in float maxRange ) {\n    return vec4( value.rgb * ( ( maxRange / 255.0 ) / value.a ), 1.0 );\n}\nvec4 LinearToRGBD( in vec4 value, in float maxRange ) {\n    float maxRGB = max( value.x, max( value.g, value.b ) );\n    float D      = max( maxRange / maxRGB, 1.0 );\n    D            = min( floor( D ) / 255.0, 1.0 );\n    return vec4( value.rgb * ( D * ( 255.0 / maxRange ) ), D );\n}\nconst mat3 cLogLuvM = mat3( 0.2209, 0.3390, 0.4184, 0.1138, 0.6780, 0.7319, 0.0102, 0.1130, 0.2969 );\nvec4 LinearToLogLuv( in vec4 value )  {\n  vec3 Xp_Y_XYZp = value.rgb * cLogLuvM;\n  Xp_Y_XYZp = max(Xp_Y_XYZp, vec3(1e-6, 1e-6, 1e-6));\n  vec4 vResult;\n  vResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;\n  float Le = 2.0 * log2(Xp_Y_XYZp.y) + 127.0;\n  vResult.w = fract(Le);\n  vResult.z = (Le - (floor(vResult.w*255.0))/255.0)/255.0;\n  return vResult;\n}\nconst mat3 cLogLuvInverseM = mat3( 6.0014, -2.7008, -1.7996, -1.3320, 3.1029, -5.7721, 0.3008, -1.0882, 5.6268 );\nvec4 LogLuvToLinear( in vec4 value ) {\n  float Le = value.z * 255.0 + value.w;\n  vec3 Xp_Y_XYZp;\n  Xp_Y_XYZp.y = exp2((Le - 127.0) / 2.0);\n  Xp_Y_XYZp.z = Xp_Y_XYZp.y / value.y;\n  Xp_Y_XYZp.x = value.x * Xp_Y_XYZp.z;\n  vec3 vRGB = Xp_Y_XYZp.rgb * cLogLuvInverseM;\n  return vec4( max(vRGB, 0.0), 1.0 );\n}\n",
n.ShaderChunk.encodings_fragment = "  gl_FragColor = linearToOutputTexel( gl_FragColor );\n",
n.ShaderChunk.envmap_fragment = "#ifdef USE_ENVMAP\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG )\n\t\tvec3 cameraToVertex = normalize( vWorldPosition - cameraPosition );\n\t\tvec3 worldNormal = inverseTransformDirection( normal, viewMatrix );\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\t\t\tvec3 reflectVec = reflect( cameraToVertex, worldNormal );\n\t\t#else\n\t\t\tvec3 reflectVec = refract( cameraToVertex, worldNormal, refractionRatio );\n\t\t#endif\n\t#else\n\t\tvec3 reflectVec = vReflect;\n\t#endif\n\t#ifdef DOUBLE_SIDED\n\t\tfloat flipNormal = ( float( gl_FrontFacing ) * 2.0 - 1.0 );\n\t#else\n\t\tfloat flipNormal = 1.0;\n\t#endif\n\t#ifdef ENVMAP_TYPE_CUBE\n\t\tvec4 envColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );\n\t#elif defined( ENVMAP_TYPE_EQUIREC )\n\t\tvec2 sampleUV;\n\t\tsampleUV.y = saturate( flipNormal * reflectVec.y * 0.5 + 0.5 );\n\t\tsampleUV.x = atan( flipNormal * reflectVec.z, flipNormal * reflectVec.x ) * RECIPROCAL_PI2 + 0.5;\n\t\tvec4 envColor = texture2D( envMap, sampleUV );\n\t#elif defined( ENVMAP_TYPE_SPHERE )\n\t\tvec3 reflectView = flipNormal * normalize((viewMatrix * vec4( reflectVec, 0.0 )).xyz + vec3(0.0,0.0,1.0));\n\t\tvec4 envColor = texture2D( envMap, reflectView.xy * 0.5 + 0.5 );\n\t#endif\n\tenvColor = envMapTexelToLinear( envColor );\n\t#ifdef ENVMAP_BLENDING_MULTIPLY\n\t\toutgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );\n\t#elif defined( ENVMAP_BLENDING_MIX )\n\t\toutgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );\n\t#elif defined( ENVMAP_BLENDING_ADD )\n\t\toutgoingLight += envColor.xyz * specularStrength * reflectivity;\n\t#endif\n#endif\n",
n.ShaderChunk.envmap_pars_fragment = "#if defined( USE_ENVMAP ) || defined( STANDARD )\n\tuniform float reflectivity;\n\tuniform float envMapIntenstiy;\n#endif\n#ifdef USE_ENVMAP\n\t#ifdef ENVMAP_TYPE_CUBE\n\t\tuniform samplerCube envMap;\n\t#else\n\t\tuniform sampler2D envMap;\n\t#endif\n\tuniform float flipEnvMap;\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( STANDARD )\n\t\tuniform float refractionRatio;\n\t#else\n\t\tvarying vec3 vReflect;\n\t#endif\n#endif\n",
n.ShaderChunk.envmap_pars_vertex = "#if defined( USE_ENVMAP ) && ! defined( USE_BUMPMAP ) && ! defined( USE_NORMALMAP ) && ! defined( PHONG ) && ! defined( STANDARD )\n\tvarying vec3 vReflect;\n\tuniform float refractionRatio;\n#endif\n",
n.ShaderChunk.envmap_vertex = "#if defined( USE_ENVMAP ) && ! defined( USE_BUMPMAP ) && ! defined( USE_NORMALMAP ) && ! defined( PHONG ) && ! defined( STANDARD )\n\tvec3 cameraToVertex = normalize( worldPosition.xyz - cameraPosition );\n\tvec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );\n\t#ifdef ENVMAP_MODE_REFLECTION\n\t\tvReflect = reflect( cameraToVertex, worldNormal );\n\t#else\n\t\tvReflect = refract( cameraToVertex, worldNormal, refractionRatio );\n\t#endif\n#endif\n",
n.ShaderChunk.fog_fragment = "#ifdef USE_FOG\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tfloat depth = gl_FragDepthEXT / gl_FragCoord.w;\n\t#else\n\t\tfloat depth = gl_FragCoord.z / gl_FragCoord.w;\n\t#endif\n\t#ifdef FOG_EXP2\n\t\tfloat fogFactor = whiteCompliment( exp2( - fogDensity * fogDensity * depth * depth * LOG2 ) );\n\t#else\n\t\tfloat fogFactor = smoothstep( fogNear, fogFar, depth );\n\t#endif\n\tgl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );\n#endif\n",
n.ShaderChunk.fog_pars_fragment = "#ifdef USE_FOG\n\tuniform vec3 fogColor;\n\t#ifdef FOG_EXP2\n\t\tuniform float fogDensity;\n\t#else\n\t\tuniform float fogNear;\n\t\tuniform float fogFar;\n\t#endif\n#endif",
n.ShaderChunk.lightmap_fragment = "#ifdef USE_LIGHTMAP\n\treflectedLight.indirectDiffuse += PI * texture2D( lightMap, vUv2 ).xyz * lightMapIntensity;\n#endif\n",
n.ShaderChunk.lightmap_pars_fragment = "#ifdef USE_LIGHTMAP\n\tuniform sampler2D lightMap;\n\tuniform float lightMapIntensity;\n#endif",
n.ShaderChunk.lights_lambert_vertex = "vec3 diffuse = vec3( 1.0 );\nGeometricContext geometry;\ngeometry.position = mvPosition.xyz;\ngeometry.normal = normalize( transformedNormal );\ngeometry.viewDir = normalize( -mvPosition.xyz );\nGeometricContext backGeometry;\nbackGeometry.position = geometry.position;\nbackGeometry.normal = -geometry.normal;\nbackGeometry.viewDir = geometry.viewDir;\nvLightFront = vec3( 0.0 );\n#ifdef DOUBLE_SIDED\n\tvLightBack = vec3( 0.0 );\n#endif\nIncidentLight directLight;\nfloat dotNL;\nvec3 directLightColor_Diffuse;\n#if NUM_POINT_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tdirectLight = getPointDirectLightIrradiance( pointLights[ i ], geometry );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = PI * directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( -dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n#endif\n#if NUM_SPOT_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tdirectLight = getSpotDirectLightIrradiance( spotLights[ i ], geometry );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = PI * directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( -dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n#endif\n#if NUM_DIR_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tdirectLight = getDirectionalDirectLightIrradiance( directionalLights[ i ], geometry );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = PI * directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( -dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n#endif\n#if NUM_HEMI_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\t\tvLightFront += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry );\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += getHemisphereLightIrradiance( hemisphereLights[ i ], backGeometry );\n\t\t#endif\n\t}\n#endif\n",
n.ShaderChunk.lights_pars = "uniform vec3 ambientLightColor;\nvec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {\n\tvec3 irradiance = ambientLightColor;\n\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\tirradiance *= PI;\n\t#endif\n\treturn irradiance;\n}\n#if NUM_DIR_LIGHTS > 0\n\tstruct DirectionalLight {\n\t\tvec3 direction;\n\t\tvec3 color;\n\t\tint shadow;\n\t\tfloat shadowBias;\n\t\tfloat shadowRadius;\n\t\tvec2 shadowMapSize;\n\t};\n\tuniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];\n\tIncidentLight getDirectionalDirectLightIrradiance( const in DirectionalLight directionalLight, const in GeometricContext geometry ) {\n\t\tIncidentLight directLight;\n\t\tdirectLight.color = directionalLight.color;\n\t\tdirectLight.direction = directionalLight.direction;\n\t\tdirectLight.visible = true;\n\t\treturn directLight;\n\t}\n#endif\n#if NUM_POINT_LIGHTS > 0\n\tstruct PointLight {\n\t\tvec3 position;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t\tint shadow;\n\t\tfloat shadowBias;\n\t\tfloat shadowRadius;\n\t\tvec2 shadowMapSize;\n\t};\n\tuniform PointLight pointLights[ NUM_POINT_LIGHTS ];\n\tIncidentLight getPointDirectLightIrradiance( const in PointLight pointLight, const in GeometricContext geometry ) {\n\t\tIncidentLight directLight;\n\t\tvec3 lVector = pointLight.position - geometry.position;\n\t\tdirectLight.direction = normalize( lVector );\n\t\tfloat lightDistance = length( lVector );\n\t\tif ( testLightInRange( lightDistance, pointLight.distance ) ) {\n\t\t\tdirectLight.color = pointLight.color;\n\t\t\tdirectLight.color *= punctualLightIntensityToIrradianceFactor( lightDistance, pointLight.distance, pointLight.decay );\n\t\t\tdirectLight.visible = true;\n\t\t} else {\n\t\t\tdirectLight.color = vec3( 0.0 );\n\t\t\tdirectLight.visible = false;\n\t\t}\n\t\treturn directLight;\n\t}\n#endif\n#if NUM_SPOT_LIGHTS > 0\n\tstruct SpotLight {\n\t\tvec3 position;\n\t\tvec3 direction;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t\tfloat coneCos;\n\t\tfloat penumbraCos;\n\t\tint shadow;\n\t\tfloat shadowBias;\n\t\tfloat shadowRadius;\n\t\tvec2 shadowMapSize;\n\t};\n\tuniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];\n\tIncidentLight getSpotDirectLightIrradiance( const in SpotLight spotLight, const in GeometricContext geometry ) {\n\t\tIncidentLight directLight;\n\t\tvec3 lVector = spotLight.position - geometry.position;\n\t\tdirectLight.direction = normalize( lVector );\n\t\tfloat lightDistance = length( lVector );\n\t\tfloat angleCos = dot( directLight.direction, spotLight.direction );\n\t\tif ( all( bvec2( angleCos > spotLight.coneCos, testLightInRange( lightDistance, spotLight.distance ) ) ) ) {\n\t\t\tfloat spotEffect = smoothstep( spotLight.coneCos, spotLight.penumbraCos, angleCos );\n\t\t\tdirectLight.color = spotLight.color;\n\t\t\tdirectLight.color *= spotEffect * punctualLightIntensityToIrradianceFactor( lightDistance, spotLight.distance, spotLight.decay );\n\t\t\tdirectLight.visible = true;\n\t\t} else {\n\t\t\tdirectLight.color = vec3( 0.0 );\n\t\t\tdirectLight.visible = false;\n\t\t}\n\t\treturn directLight;\n\t}\n#endif\n#if NUM_HEMI_LIGHTS > 0\n\tstruct HemisphereLight {\n\t\tvec3 direction;\n\t\tvec3 skyColor;\n\t\tvec3 groundColor;\n\t};\n\tuniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];\n\tvec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in GeometricContext geometry ) {\n\t\tfloat dotNL = dot( geometry.normal, hemiLight.direction );\n\t\tfloat hemiDiffuseWeight = 0.5 * dotNL + 0.5;\n\t\tvec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );\n\t\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\t\tirradiance *= PI;\n\t\t#endif\n\t\treturn irradiance;\n\t}\n#endif\n#if defined( USE_ENVMAP ) && defined( STANDARD )\n\tvec3 getLightProbeIndirectIrradiance( const in GeometricContext geometry, const in int maxMIPLevel ) {\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tfloat flipNormal = ( float( gl_FrontFacing ) * 2.0 - 1.0 );\n\t\t#else\n\t\t\tfloat flipNormal = 1.0;\n\t\t#endif\n\t\tvec3 worldNormal = inverseTransformDirection( geometry.normal, viewMatrix );\n\t\t#ifdef ENVMAP_TYPE_CUBE\n\t\t\tvec3 queryVec = flipNormal * vec3( flipEnvMap * worldNormal.x, worldNormal.yz );\n\t\t\t#ifdef TEXTURE_LOD_EXT\n\t\t\t\tvec4 envMapColor = textureCubeLodEXT( envMap, queryVec, float( maxMIPLevel ) );\n\t\t\t#else\n\t\t\t\tvec4 envMapColor = textureCube( envMap, queryVec, float( maxMIPLevel ) );\n\t\t\t#endif\n\t\t#elif defined( ENVMAP_TYPE_CUBE_UV )\n\t\t\tvec3 queryVec = flipNormal * vec3( flipEnvMap * worldNormal.x, worldNormal.yz );\n\t\t\tvec4 envMapColor = textureCubeUV( queryVec, 1.0 );\n\t\t#else\n\t\t\tvec4 envMapColor = vec4( 0.0 );\n\t\t#endif\n\t\tenvMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;\n\t\treturn PI * envMapColor.rgb * envMapIntensity;\n\t}\n\tfloat getSpecularMIPLevel( const in float blinnShininessExponent, const in int maxMIPLevel ) {\n\t\tfloat maxMIPLevelScalar = float( maxMIPLevel );\n\t\tfloat desiredMIPLevel = maxMIPLevelScalar - 0.79248 - 0.5 * log2( pow2( blinnShininessExponent ) + 1.0 );\n\t\treturn clamp( desiredMIPLevel, 0.0, maxMIPLevelScalar );\n\t}\n\tvec3 getLightProbeIndirectRadiance( const in GeometricContext geometry, const in float blinnShininessExponent, const in int maxMIPLevel ) {\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\t\t\tvec3 reflectVec = reflect( -geometry.viewDir, geometry.normal );\n\t\t#else\n\t\t\tvec3 reflectVec = refract( -geometry.viewDir, geometry.normal, refractionRatio );\n\t\t#endif\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tfloat flipNormal = ( float( gl_FrontFacing ) * 2.0 - 1.0 );\n\t\t#else\n\t\t\tfloat flipNormal = 1.0;\n\t\t#endif\n\t\treflectVec = inverseTransformDirection( reflectVec, viewMatrix );\n\t\tfloat specularMIPLevel = getSpecularMIPLevel( blinnShininessExponent, maxMIPLevel );\n\t\t#ifdef ENVMAP_TYPE_CUBE\n\t\t\tvec3 queryReflectVec = flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz );\n\t\t\t#ifdef TEXTURE_LOD_EXT\n\t\t\t\tvec4 envMapColor = textureCubeLodEXT( envMap, queryReflectVec, specularMIPLevel );\n\t\t\t#else\n\t\t\t\tvec4 envMapColor = textureCube( envMap, queryReflectVec, specularMIPLevel );\n\t\t\t#endif\n\t\t#elif defined( ENVMAP_TYPE_CUBE_UV )\n\t\t\tvec3 queryReflectVec = flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz );\n\t\t\tvec4 envMapColor = textureCubeUV(queryReflectVec, BlinnExponentToGGXRoughness(blinnShininessExponent));\n\t\t#elif defined( ENVMAP_TYPE_EQUIREC )\n\t\t\tvec2 sampleUV;\n\t\t\tsampleUV.y = saturate( flipNormal * reflectVec.y * 0.5 + 0.5 );\n\t\t\tsampleUV.x = atan( flipNormal * reflectVec.z, flipNormal * reflectVec.x ) * RECIPROCAL_PI2 + 0.5;\n\t\t\t#ifdef TEXTURE_LOD_EXT\n\t\t\t\tvec4 envMapColor = texture2DLodEXT( envMap, sampleUV, specularMIPLevel );\n\t\t\t#else\n\t\t\t\tvec4 envMapColor = texture2D( envMap, sampleUV, specularMIPLevel );\n\t\t\t#endif\n\t\t#elif defined( ENVMAP_TYPE_SPHERE )\n\t\t\tvec3 reflectView = flipNormal * normalize((viewMatrix * vec4( reflectVec, 0.0 )).xyz + vec3(0.0,0.0,1.0));\n\t\t\t#ifdef TEXTURE_LOD_EXT\n\t\t\t\tvec4 envMapColor = texture2DLodEXT( envMap, reflectView.xy * 0.5 + 0.5, specularMIPLevel );\n\t\t\t#else\n\t\t\t\tvec4 envMapColor = texture2D( envMap, reflectView.xy * 0.5 + 0.5, specularMIPLevel );\n\t\t\t#endif\n\t\t#endif\n\t\tenvMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;\n\t\treturn envMapColor.rgb * envMapIntensity;\n\t}\n#endif\n",
n.ShaderChunk.lights_phong_fragment = "BlinnPhongMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularColor = specular;\nmaterial.specularShininess = shininess;\nmaterial.specularStrength = specularStrength;\n",
n.ShaderChunk.lights_phong_pars_fragment = "#ifdef USE_ENVMAP\n\tvarying vec3 vWorldPosition;\n#endif\nvarying vec3 vViewPosition;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\nstruct BlinnPhongMaterial {\n\tvec3\tdiffuseColor;\n\tvec3\tspecularColor;\n\tfloat\tspecularShininess;\n\tfloat\tspecularStrength;\n};\nvoid RE_Direct_BlinnPhong( const in IncidentLight directLight, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\tfloat dotNL = saturate( dot( geometry.normal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\tirradiance *= PI;\n\t#endif\n\treflectedLight.directDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );\n\treflectedLight.directSpecular += irradiance * BRDF_Specular_BlinnPhong( directLight, geometry, material.specularColor, material.specularShininess ) * material.specularStrength;\n}\nvoid RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );\n}\n#define RE_Direct\t\t\t\tRE_Direct_BlinnPhong\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_BlinnPhong\n#define Material_LightProbeLOD( material )\t(0)\n",
n.ShaderChunk.lights_phong_pars_vertex = "#ifdef USE_ENVMAP\n\tvarying vec3 vWorldPosition;\n#endif\n",
n.ShaderChunk.lights_phong_vertex = "#ifdef USE_ENVMAP\n\tvWorldPosition = worldPosition.xyz;\n#endif\n",
n.ShaderChunk.lights_standard_fragment = "StandardMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );\nmaterial.specularRoughness = clamp( roughnessFactor, 0.04, 1.0 );\nmaterial.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );\n",
n.ShaderChunk.lights_standard_pars_fragment = "struct StandardMaterial {\n\tvec3\tdiffuseColor;\n\tfloat\tspecularRoughness;\n\tvec3\tspecularColor;\n};\nvoid RE_Direct_Standard( const in IncidentLight directLight, const in GeometricContext geometry, const in StandardMaterial material, inout ReflectedLight reflectedLight ) {\n\tfloat dotNL = saturate( dot( geometry.normal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\tirradiance *= PI;\n\t#endif\n\treflectedLight.directDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );\n\treflectedLight.directSpecular += irradiance * BRDF_Specular_GGX( directLight, geometry, material.specularColor, material.specularRoughness );\n}\nvoid RE_IndirectDiffuse_Standard( const in vec3 irradiance, const in GeometricContext geometry, const in StandardMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectSpecular_Standard( const in vec3 radiance, const in GeometricContext geometry, const in StandardMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectSpecular += radiance * BRDF_Specular_GGX_Environment( geometry, material.specularColor, material.specularRoughness );\n}\n#define RE_Direct\t\t\t\tRE_Direct_Standard\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Standard\n#define RE_IndirectSpecular\t\tRE_IndirectSpecular_Standard\n#define Material_BlinnShininessExponent( material )   GGXRoughnessToBlinnExponent( material.specularRoughness )\nfloat computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {\n\treturn saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );\n}\n",
n.ShaderChunk.lights_template = "\nGeometricContext geometry;\ngeometry.position = - vViewPosition;\ngeometry.normal = normal;\ngeometry.viewDir = normalize( vViewPosition );\nIncidentLight directLight;\n#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )\n\tPointLight pointLight;\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tpointLight = pointLights[ i ];\n\t\tdirectLight = getPointDirectLightIrradiance( pointLight, geometry );\n\t\t#ifdef USE_SHADOWMAP\n\t\tdirectLight.color *= all( bvec2( pointLight.shadow, directLight.visible ) ) ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n#endif\n#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )\n\tSpotLight spotLight;\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tspotLight = spotLights[ i ];\n\t\tdirectLight = getSpotDirectLightIrradiance( spotLight, geometry );\n\t\t#ifdef USE_SHADOWMAP\n\t\tdirectLight.color *= all( bvec2( spotLight.shadow, directLight.visible ) ) ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowBias, spotLight.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n#endif\n#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )\n\tDirectionalLight directionalLight;\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tdirectionalLight = directionalLights[ i ];\n\t\tdirectLight = getDirectionalDirectLightIrradiance( directionalLight, geometry );\n\t\t#ifdef USE_SHADOWMAP\n\t\tdirectLight.color *= all( bvec2( directionalLight.shadow, directLight.visible ) ) ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n#endif\n#if defined( RE_IndirectDiffuse )\n\tvec3 irradiance = getAmbientLightIrradiance( ambientLightColor );\n\t#ifdef USE_LIGHTMAP\n\t\tvec3 lightMapIrradiance = texture2D( lightMap, vUv2 ).xyz * lightMapIntensity;\n\t\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\t\tlightMapIrradiance *= PI;\n\t\t#endif\n\t\tirradiance += lightMapIrradiance;\n\t#endif\n\t#if ( NUM_HEMI_LIGHTS > 0 )\n\t\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\t\t\tirradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry );\n\t\t}\n\t#endif\n\t#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )\n\t \tirradiance += getLightProbeIndirectIrradiance( geometry, 8 );\n\t#endif\n\tRE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );\n#endif\n#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )\n\tvec3 radiance = getLightProbeIndirectRadiance( geometry, Material_BlinnShininessExponent( material ), 8 );\n\tRE_IndirectSpecular( radiance, geometry, material, reflectedLight );\n#endif\n",
n.ShaderChunk.logdepthbuf_fragment = "#if defined(USE_LOGDEPTHBUF) && defined(USE_LOGDEPTHBUF_EXT)\n\tgl_FragDepthEXT = log2(vFragDepth) * logDepthBufFC * 0.5;\n#endif",
n.ShaderChunk.logdepthbuf_pars_fragment = "#ifdef USE_LOGDEPTHBUF\n\tuniform float logDepthBufFC;\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tvarying float vFragDepth;\n\t#endif\n#endif\n",
n.ShaderChunk.logdepthbuf_pars_vertex = "#ifdef USE_LOGDEPTHBUF\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tvarying float vFragDepth;\n\t#endif\n\tuniform float logDepthBufFC;\n#endif",
n.ShaderChunk.logdepthbuf_vertex = "#ifdef USE_LOGDEPTHBUF\n\tgl_Position.z = log2(max( EPSILON, gl_Position.w + 1.0 )) * logDepthBufFC;\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tvFragDepth = 1.0 + gl_Position.w;\n\t#else\n\t\tgl_Position.z = (gl_Position.z - 1.0) * gl_Position.w;\n\t#endif\n#endif\n",
n.ShaderChunk.map_fragment = "#ifdef USE_MAP\n\tvec4 texelColor = texture2D( map, vUv );\n\ttexelColor = mapTexelToLinear( texelColor );\n\tdiffuseColor *= texelColor;\n#endif\n",
n.ShaderChunk.map_pars_fragment = "#ifdef USE_MAP\n\tuniform sampler2D map;\n#endif\n",
n.ShaderChunk.map_particle_fragment = "#ifdef USE_MAP\n\tvec4 mapTexel = texture2D( map, vec2( gl_PointCoord.x, 1.0 - gl_PointCoord.y ) * offsetRepeat.zw + offsetRepeat.xy );\n\tdiffuseColor *= mapTexelToLinear( mapTexel );\n#endif\n",
n.ShaderChunk.map_particle_pars_fragment = "#ifdef USE_MAP\n\tuniform vec4 offsetRepeat;\n\tuniform sampler2D map;\n#endif\n",
n.ShaderChunk.metalnessmap_fragment = "float metalnessFactor = metalness;\n#ifdef USE_METALNESSMAP\n\tvec4 texelMetalness = texture2D( metalnessMap, vUv );\n\tmetalnessFactor *= texelMetalness.r;\n#endif\n",
n.ShaderChunk.metalnessmap_pars_fragment = "#ifdef USE_METALNESSMAP\n\tuniform sampler2D metalnessMap;\n#endif",
n.ShaderChunk.morphnormal_vertex = "#ifdef USE_MORPHNORMALS\n\tobjectNormal += ( morphNormal0 - normal ) * morphTargetInfluences[ 0 ];\n\tobjectNormal += ( morphNormal1 - normal ) * morphTargetInfluences[ 1 ];\n\tobjectNormal += ( morphNormal2 - normal ) * morphTargetInfluences[ 2 ];\n\tobjectNormal += ( morphNormal3 - normal ) * morphTargetInfluences[ 3 ];\n#endif\n",
n.ShaderChunk.morphtarget_pars_vertex = "#ifdef USE_MORPHTARGETS\n\t#ifndef USE_MORPHNORMALS\n\tuniform float morphTargetInfluences[ 8 ];\n\t#else\n\tuniform float morphTargetInfluences[ 4 ];\n\t#endif\n#endif";
n.ShaderChunk.morphtarget_vertex = "#ifdef USE_MORPHTARGETS\n\ttransformed += ( morphTarget0 - position ) * morphTargetInfluences[ 0 ];\n\ttransformed += ( morphTarget1 - position ) * morphTargetInfluences[ 1 ];\n\ttransformed += ( morphTarget2 - position ) * morphTargetInfluences[ 2 ];\n\ttransformed += ( morphTarget3 - position ) * morphTargetInfluences[ 3 ];\n\t#ifndef USE_MORPHNORMALS\n\ttransformed += ( morphTarget4 - position ) * morphTargetInfluences[ 4 ];\n\ttransformed += ( morphTarget5 - position ) * morphTargetInfluences[ 5 ];\n\ttransformed += ( morphTarget6 - position ) * morphTargetInfluences[ 6 ];\n\ttransformed += ( morphTarget7 - position ) * morphTargetInfluences[ 7 ];\n\t#endif\n#endif\n";
n.ShaderChunk.normal_fragment = "#ifdef FLAT_SHADED\n\tvec3 fdx = vec3( dFdx( vViewPosition.x ), dFdx( vViewPosition.y ), dFdx( vViewPosition.z ) );\n\tvec3 fdy = vec3( dFdy( vViewPosition.x ), dFdy( vViewPosition.y ), dFdy( vViewPosition.z ) );\n\tvec3 normal = normalize( cross( fdx, fdy ) );\n#else\n\tvec3 normal = normalize( vNormal );\n\t#ifdef DOUBLE_SIDED\n\t\tnormal = normal * ( -1.0 + 2.0 * float( gl_FrontFacing ) );\n\t#endif\n#endif\n#ifdef USE_NORMALMAP\n\tnormal = perturbNormal2Arb( -vViewPosition, normal );\n#elif defined( USE_BUMPMAP )\n\tnormal = perturbNormalArb( -vViewPosition, normal, dHdxy_fwd() );\n#endif\n",
n.ShaderChunk.normalmap_pars_fragment = "#ifdef USE_NORMALMAP\n\tuniform sampler2D normalMap;\n\tuniform vec2 normalScale;\n\tvec3 perturbNormal2Arb( vec3 eye_pos, vec3 surf_norm ) {\n\t\tvec3 q0 = dFdx( eye_pos.xyz );\n\t\tvec3 q1 = dFdy( eye_pos.xyz );\n\t\tvec2 st0 = dFdx( vUv.st );\n\t\tvec2 st1 = dFdy( vUv.st );\n\t\tvec3 S = normalize( q0 * st1.t - q1 * st0.t );\n\t\tvec3 T = normalize( -q0 * st1.s + q1 * st0.s );\n\t\tvec3 N = normalize( surf_norm );\n\t\tvec3 mapN = texture2D( normalMap, vUv ).xyz * 2.0 - 1.0;\n\t\tmapN.xy = normalScale * mapN.xy;\n\t\tmat3 tsn = mat3( S, T, N );\n\t\treturn normalize( tsn * mapN );\n\t}\n#endif\n",
n.ShaderChunk.premultiplied_alpha_fragment = "#ifdef PREMULTIPLIED_ALPHA\n\tgl_FragColor.rgb *= gl_FragColor.a;\n#endif\n",
n.ShaderChunk.project_vertex = "#ifdef USE_SKINNING\n\tvec4 mvPosition = modelViewMatrix * skinned;\n#else\n\tvec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );\n#endif\ngl_Position = projectionMatrix * mvPosition;\n",
n.ShaderChunk.roughnessmap_fragment = "float roughnessFactor = roughness;\n#ifdef USE_ROUGHNESSMAP\n\tvec4 texelRoughness = texture2D( roughnessMap, vUv );\n\troughnessFactor *= texelRoughness.r;\n#endif\n",
n.ShaderChunk.roughnessmap_pars_fragment = "#ifdef USE_ROUGHNESSMAP\n\tuniform sampler2D roughnessMap;\n#endif",
n.ShaderChunk.shadowmap_pars_fragment = "#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHTS > 0\n\t\tuniform sampler2D directionalShadowMap[ NUM_DIR_LIGHTS ];\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHTS ];\n\t#endif\n\t#if NUM_SPOT_LIGHTS > 0\n\t\tuniform sampler2D spotShadowMap[ NUM_SPOT_LIGHTS ];\n\t\tvarying vec4 vSpotShadowCoord[ NUM_SPOT_LIGHTS ];\n\t#endif\n\t#if NUM_POINT_LIGHTS > 0\n\t\tuniform sampler2D pointShadowMap[ NUM_POINT_LIGHTS ];\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHTS ];\n\t#endif\n\tfloat unpackDepth( const in vec4 rgba_depth ) {\n\t\tconst vec4 bit_shift = vec4( 1.0 / ( 256.0 * 256.0 * 256.0 ), 1.0 / ( 256.0 * 256.0 ), 1.0 / 256.0, 1.0 );\n\t\treturn dot( rgba_depth, bit_shift );\n\t}\n\tfloat texture2DCompare( sampler2D depths, vec2 uv, float compare ) {\n\t\treturn step( compare, unpackDepth( texture2D( depths, uv ) ) );\n\t}\n\tfloat texture2DShadowLerp( sampler2D depths, vec2 size, vec2 uv, float compare ) {\n\t\tconst vec2 offset = vec2( 0.0, 1.0 );\n\t\tvec2 texelSize = vec2( 1.0 ) / size;\n\t\tvec2 centroidUV = floor( uv * size + 0.5 ) / size;\n\t\tfloat lb = texture2DCompare( depths, centroidUV + texelSize * offset.xx, compare );\n\t\tfloat lt = texture2DCompare( depths, centroidUV + texelSize * offset.xy, compare );\n\t\tfloat rb = texture2DCompare( depths, centroidUV + texelSize * offset.yx, compare );\n\t\tfloat rt = texture2DCompare( depths, centroidUV + texelSize * offset.yy, compare );\n\t\tvec2 f = fract( uv * size + 0.5 );\n\t\tfloat a = mix( lb, lt, f.y );\n\t\tfloat b = mix( rb, rt, f.y );\n\t\tfloat c = mix( a, b, f.x );\n\t\treturn c;\n\t}\n\tfloat getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\t\tshadowCoord.xyz /= shadowCoord.w;\n\t\tshadowCoord.z += shadowBias;\n\t\tbvec4 inFrustumVec = bvec4 ( shadowCoord.x >= 0.0, shadowCoord.x <= 1.0, shadowCoord.y >= 0.0, shadowCoord.y <= 1.0 );\n\t\tbool inFrustum = all( inFrustumVec );\n\t\tbvec2 frustumTestVec = bvec2( inFrustum, shadowCoord.z <= 1.0 );\n\t\tbool frustumTest = all( frustumTestVec );\n\t\tif ( frustumTest ) {\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\t\t\tvec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n\t\t\tfloat dx0 = - texelSize.x * shadowRadius;\n\t\t\tfloat dy0 = - texelSize.y * shadowRadius;\n\t\t\tfloat dx1 = + texelSize.x * shadowRadius;\n\t\t\tfloat dy1 = + texelSize.y * shadowRadius;\n\t\t\treturn (\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )\n\t\t\t) * ( 1.0 / 9.0 );\n\t\t#elif defined( SHADOWMAP_TYPE_PCF_SOFT )\n\t\t\tvec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n\t\t\tfloat dx0 = - texelSize.x * shadowRadius;\n\t\t\tfloat dy0 = - texelSize.y * shadowRadius;\n\t\t\tfloat dx1 = + texelSize.x * shadowRadius;\n\t\t\tfloat dy1 = + texelSize.y * shadowRadius;\n\t\t\treturn (\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy, shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )\n\t\t\t) * ( 1.0 / 9.0 );\n\t\t#else\n\t\t\treturn texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );\n\t\t#endif\n\t\t}\n\t\treturn 1.0;\n\t}\n\tvec2 cubeToUV( vec3 v, float texelSizeY ) {\n\t\tvec3 absV = abs( v );\n\t\tfloat scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );\n\t\tabsV *= scaleToCube;\n\t\tv *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );\n\t\tvec2 planar = v.xy;\n\t\tfloat almostATexel = 1.5 * texelSizeY;\n\t\tfloat almostOne = 1.0 - almostATexel;\n\t\tif ( absV.z >= almostOne ) {\n\t\t\tif ( v.z > 0.0 )\n\t\t\t\tplanar.x = 4.0 - v.x;\n\t\t} else if ( absV.x >= almostOne ) {\n\t\t\tfloat signX = sign( v.x );\n\t\t\tplanar.x = v.z * signX + 2.0 * signX;\n\t\t} else if ( absV.y >= almostOne ) {\n\t\t\tfloat signY = sign( v.y );\n\t\t\tplanar.x = v.x + 2.0 * signY + 2.0;\n\t\t\tplanar.y = v.z * signY - 2.0;\n\t\t}\n\t\treturn vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );\n\t}\n\tfloat getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\t\tvec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );\n\t\tvec3 lightToPosition = shadowCoord.xyz;\n\t\tvec3 bd3D = normalize( lightToPosition );\n\t\tfloat dp = ( length( lightToPosition ) - shadowBias ) / 1000.0;\n\t\t#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT )\n\t\t\tvec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;\n\t\t\treturn (\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )\n\t\t\t) * ( 1.0 / 9.0 );\n\t\t#else\n\t\t\treturn texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );\n\t\t#endif\n\t}\n#endif\n",
n.ShaderChunk.shadowmap_pars_vertex = "#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHTS > 0\n\t\tuniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHTS ];\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHTS ];\n\t#endif\n\t#if NUM_SPOT_LIGHTS > 0\n\t\tuniform mat4 spotShadowMatrix[ NUM_SPOT_LIGHTS ];\n\t\tvarying vec4 vSpotShadowCoord[ NUM_SPOT_LIGHTS ];\n\t#endif\n\t#if NUM_POINT_LIGHTS > 0\n\t\tuniform mat4 pointShadowMatrix[ NUM_POINT_LIGHTS ];\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHTS ];\n\t#endif\n#endif\n",
n.ShaderChunk.shadowmap_vertex = "#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tvDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * worldPosition;\n\t}\n\t#endif\n\t#if NUM_SPOT_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tvSpotShadowCoord[ i ] = spotShadowMatrix[ i ] * worldPosition;\n\t}\n\t#endif\n\t#if NUM_POINT_LIGHTS > 0\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tvPointShadowCoord[ i ] = pointShadowMatrix[ i ] * worldPosition;\n\t}\n\t#endif\n#endif\n",
n.ShaderChunk.shadowmask_pars_fragment = "float getShadowMask() {\n\tfloat shadow = 1.0;\n\t#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHTS > 0\n\tDirectionalLight directionalLight;\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tdirectionalLight = directionalLights[ i ];\n\t\tshadow *= bool( directionalLight.shadow ) ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t}\n\t#endif\n\t#if NUM_SPOT_LIGHTS > 0\n\tSpotLight spotLight;\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tspotLight = spotLights[ i ];\n\t\tshadow *= bool( spotLight.shadow ) ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowBias, spotLight.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;\n\t}\n\t#endif\n\t#if NUM_POINT_LIGHTS > 0\n\tPointLight pointLight;\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tpointLight = pointLights[ i ];\n\t\tshadow *= bool( pointLight.shadow ) ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ] ) : 1.0;\n\t}\n\t#endif\n\t#endif\n\treturn shadow;\n}\n",
n.ShaderChunk.skinbase_vertex = "#ifdef USE_SKINNING\n\tmat4 boneMatX = getBoneMatrix( skinIndex.x );\n\tmat4 boneMatY = getBoneMatrix( skinIndex.y );\n\tmat4 boneMatZ = getBoneMatrix( skinIndex.z );\n\tmat4 boneMatW = getBoneMatrix( skinIndex.w );\n#endif",
n.ShaderChunk.skinning_pars_vertex = "#ifdef USE_SKINNING\n\tuniform mat4 bindMatrix;\n\tuniform mat4 bindMatrixInverse;\n\t#ifdef BONE_TEXTURE\n\t\tuniform sampler2D boneTexture;\n\t\tuniform int boneTextureWidth;\n\t\tuniform int boneTextureHeight;\n\t\tmat4 getBoneMatrix( const in float i ) {\n\t\t\tfloat j = i * 4.0;\n\t\t\tfloat x = mod( j, float( boneTextureWidth ) );\n\t\t\tfloat y = floor( j / float( boneTextureWidth ) );\n\t\t\tfloat dx = 1.0 / float( boneTextureWidth );\n\t\t\tfloat dy = 1.0 / float( boneTextureHeight );\n\t\t\ty = dy * ( y + 0.5 );\n\t\t\tvec4 v1 = texture2D( boneTexture, vec2( dx * ( x + 0.5 ), y ) );\n\t\t\tvec4 v2 = texture2D( boneTexture, vec2( dx * ( x + 1.5 ), y ) );\n\t\t\tvec4 v3 = texture2D( boneTexture, vec2( dx * ( x + 2.5 ), y ) );\n\t\t\tvec4 v4 = texture2D( boneTexture, vec2( dx * ( x + 3.5 ), y ) );\n\t\t\tmat4 bone = mat4( v1, v2, v3, v4 );\n\t\t\treturn bone;\n\t\t}\n\t#else\n\t\tuniform mat4 boneGlobalMatrices[ MAX_BONES ];\n\t\tmat4 getBoneMatrix( const in float i ) {\n\t\t\tmat4 bone = boneGlobalMatrices[ int(i) ];\n\t\t\treturn bone;\n\t\t}\n\t#endif\n#endif\n",
n.ShaderChunk.skinning_vertex = "#ifdef USE_SKINNING\n\tvec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );\n\tvec4 skinned = vec4( 0.0 );\n\tskinned += boneMatX * skinVertex * skinWeight.x;\n\tskinned += boneMatY * skinVertex * skinWeight.y;\n\tskinned += boneMatZ * skinVertex * skinWeight.z;\n\tskinned += boneMatW * skinVertex * skinWeight.w;\n\tskinned  = bindMatrixInverse * skinned;\n#endif\n",
n.ShaderChunk.skinnormal_vertex = "#ifdef USE_SKINNING\n\tmat4 skinMatrix = mat4( 0.0 );\n\tskinMatrix += skinWeight.x * boneMatX;\n\tskinMatrix += skinWeight.y * boneMatY;\n\tskinMatrix += skinWeight.z * boneMatZ;\n\tskinMatrix += skinWeight.w * boneMatW;\n\tskinMatrix  = bindMatrixInverse * skinMatrix * bindMatrix;\n\tobjectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;\n#endif\n",
n.ShaderChunk.specularmap_fragment = "float specularStrength;\n#ifdef USE_SPECULARMAP\n\tvec4 texelSpecular = texture2D( specularMap, vUv );\n\tspecularStrength = texelSpecular.r;\n#else\n\tspecularStrength = 1.0;\n#endif",
n.ShaderChunk.specularmap_pars_fragment = "#ifdef USE_SPECULARMAP\n\tuniform sampler2D specularMap;\n#endif",
n.ShaderChunk.tonemapping_fragment = "#if defined( TONE_MAPPING )\n  gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );\n#endif\n",
n.ShaderChunk.tonemapping_pars_fragment = "#define saturate(a) clamp( a, 0.0, 1.0 )\nuniform float toneMappingExposure;\nuniform float toneMappingWhitePoint;\nvec3 LinearToneMapping( vec3 color ) {\n  return toneMappingExposure * color;\n}\nvec3 ReinhardToneMapping( vec3 color ) {\n  color *= toneMappingExposure;\n  return saturate( color / ( vec3( 1.0 ) + color ) );\n}\n#define Uncharted2Helper( x ) max( ( ( x * ( 0.15 * x + 0.10 * 0.50 ) + 0.20 * 0.02 ) / ( x * ( 0.15 * x + 0.50 ) + 0.20 * 0.30 ) ) - 0.02 / 0.30, vec3( 0.0 ) )\nvec3 Uncharted2ToneMapping( vec3 color ) {\n  color *= toneMappingExposure;\n  return saturate( Uncharted2Helper( color ) / Uncharted2Helper( vec3( toneMappingWhitePoint ) ) );\n}\nvec3 OptimizedCineonToneMapping( vec3 color ) {\n  color *= toneMappingExposure;\n  color = max( vec3( 0.0 ), color - 0.004 );\n  return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );\n}\n",
n.ShaderChunk.uv2_pars_fragment = "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tvarying vec2 vUv2;\n#endif",
n.ShaderChunk.uv2_pars_vertex = "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tattribute vec2 uv2;\n\tvarying vec2 vUv2;\n#endif",
n.ShaderChunk.uv2_vertex = "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tvUv2 = uv2;\n#endif",
n.ShaderChunk.uv_pars_fragment = "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )\n\tvarying vec2 vUv;\n#endif",
n.ShaderChunk.uv_pars_vertex = "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )\n\tvarying vec2 vUv;\n\tuniform vec4 offsetRepeat;\n#endif\n",
n.ShaderChunk.uv_vertex = "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )\n\tvUv = uv * offsetRepeat.zw + offsetRepeat.xy;\n#endif",
n.ShaderChunk.worldpos_vertex = "#if defined( USE_ENVMAP ) || defined( PHONG ) || defined( STANDARD ) || defined( LAMBERT ) || defined ( USE_SHADOWMAP )\n\t#ifdef USE_SKINNING\n\t\tvec4 worldPosition = modelMatrix * skinned;\n\t#else\n\t\tvec4 worldPosition = modelMatrix * vec4( transformed, 1.0 );\n\t#endif\n#endif\n",
n.UniformsUtils = {
    merge: function(e) {
        for (var t = {}, i = 0; i < e.length; i++) {
            var n = this.clone(e[i]);
            for (var r in n)
                t[r] = n[r]
        }
        return t
    },
    clone: function(e) {
        var t = {};
        for (var i in e) {
            t[i] = {};
            for (var r in e[i]) {
                var o = e[i][r];
                o instanceof n.Color || o instanceof n.Vector2 || o instanceof n.Vector3 || o instanceof n.Vector4 || o instanceof n.Matrix3 || o instanceof n.Matrix4 || o instanceof n.Texture ? t[i][r] = o.clone() : Array.isArray(o) ? t[i][r] = o.slice() : t[i][r] = o
            }
        }
        return t
    }
},
n.UniformsLib = {
    common: {
        diffuse: {
            type: "c",
            value: new n.Color(15658734)
        },
        opacity: {
            type: "f",
            value: 1
        },
        map: {
            type: "t",
            value: null
        },
        offsetRepeat: {
            type: "v4",
            value: new n.Vector4(0,0,1,1)
        },
        specularMap: {
            type: "t",
            value: null
        },
        alphaMap: {
            type: "t",
            value: null
        },
        envMap: {
            type: "t",
            value: null
        },
        flipEnvMap: {
            type: "f",
            value: -1
        },
        reflectivity: {
            type: "f",
            value: 1
        },
        refractionRatio: {
            type: "f",
            value: .98
        }
    },
    aomap: {
        aoMap: {
            type: "t",
            value: null
        },
        aoMapIntensity: {
            type: "f",
            value: 1
        }
    },
    lightmap: {
        lightMap: {
            type: "t",
            value: null
        },
        lightMapIntensity: {
            type: "f",
            value: 1
        }
    },
    emissivemap: {
        emissiveMap: {
            type: "t",
            value: null
        }
    },
    bumpmap: {
        bumpMap: {
            type: "t",
            value: null
        },
        bumpScale: {
            type: "f",
            value: 1
        }
    },
    normalmap: {
        normalMap: {
            type: "t",
            value: null
        },
        normalScale: {
            type: "v2",
            value: new n.Vector2(1,1)
        }
    },
    displacementmap: {
        displacementMap: {
            type: "t",
            value: null
        },
        displacementScale: {
            type: "f",
            value: 1
        },
        displacementBias: {
            type: "f",
            value: 0
        }
    },
    roughnessmap: {
        roughnessMap: {
            type: "t",
            value: null
        }
    },
    metalnessmap: {
        metalnessMap: {
            type: "t",
            value: null
        }
    },
    fog: {
        fogDensity: {
            type: "f",
            value: 25e-5
        },
        fogNear: {
            type: "f",
            value: 1
        },
        fogFar: {
            type: "f",
            value: 2e3
        },
        fogColor: {
            type: "c",
            value: new n.Color(16777215)
        }
    },
    lights: {
        ambientLightColor: {
            type: "fv",
            value: []
        },
        directionalLights: {
            type: "sa",
            value: [],
            properties: {
                direction: {
                    type: "v3"
                },
                color: {
                    type: "c"
                },
                shadow: {
                    type: "i"
                },
                shadowBias: {
                    type: "f"
                },
                shadowRadius: {
                    type: "f"
                },
                shadowMapSize: {
                    type: "v2"
                }
            }
        },
        directionalShadowMap: {
            type: "tv",
            value: []
        },
        directionalShadowMatrix: {
            type: "m4v",
            value: []
        },
        spotLights: {
            type: "sa",
            value: [],
            properties: {
                color: {
                    type: "c"
                },
                position: {
                    type: "v3"
                },
                direction: {
                    type: "v3"
                },
                distance: {
                    type: "f"
                },
                coneCos: {
                    type: "f"
                },
                penumbraCos: {
                    type: "f"
                },
                decay: {
                    type: "f"
                },
                shadow: {
                    type: "i"
                },
                shadowBias: {
                    type: "f"
                },
                shadowRadius: {
                    type: "f"
                },
                shadowMapSize: {
                    type: "v2"
                }
            }
        },
        spotShadowMap: {
            type: "tv",
            value: []
        },
        spotShadowMatrix: {
            type: "m4v",
            value: []
        },
        pointLights: {
            type: "sa",
            value: [],
            properties: {
                color: {
                    type: "c"
                },
                position: {
                    type: "v3"
                },
                decay: {
                    type: "f"
                },
                distance: {
                    type: "f"
                },
                shadow: {
                    type: "i"
                },
                shadowBias: {
                    type: "f"
                },
                shadowRadius: {
                    type: "f"
                },
                shadowMapSize: {
                    type: "v2"
                }
            }
        },
        pointShadowMap: {
            type: "tv",
            value: []
        },
        pointShadowMatrix: {
            type: "m4v",
            value: []
        },
        hemisphereLights: {
            type: "sa",
            value: [],
            properties: {
                direction: {
                    type: "v3"
                },
                skyColor: {
                    type: "c"
                },
                groundColor: {
                    type: "c"
                }
            }
        }
    },
    points: {
        diffuse: {
            type: "c",
            value: new n.Color(15658734)
        },
        opacity: {
            type: "f",
            value: 1
        },
        size: {
            type: "f",
            value: 1
        },
        scale: {
            type: "f",
            value: 1
        },
        map: {
            type: "t",
            value: null
        },
        offsetRepeat: {
            type: "v4",
            value: new n.Vector4(0,0,1,1)
        }
    }
},
n.ShaderChunk.cube_frag = "uniform samplerCube tCube;\nuniform float tFlip;\nvarying vec3 vWorldPosition;\n#include <common>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tgl_FragColor = textureCube( tCube, vec3( tFlip * vWorldPosition.x, vWorldPosition.yz ) );\n\t#include <logdepthbuf_fragment>\n}\n",
n.ShaderChunk.cube_vert = "varying vec3 vWorldPosition;\n#include <common>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\tvWorldPosition = transformDirection( position, modelMatrix );\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.depth_frag = "uniform float mNear;\nuniform float mFar;\nuniform float opacity;\n#include <common>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\t#include <logdepthbuf_fragment>\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tfloat depth = gl_FragDepthEXT / gl_FragCoord.w;\n\t#else\n\t\tfloat depth = gl_FragCoord.z / gl_FragCoord.w;\n\t#endif\n\tfloat color = 1.0 - smoothstep( mNear, mFar, depth );\n\tgl_FragColor = vec4( vec3( color ), opacity );\n}\n",
n.ShaderChunk.depth_vert = "#include <common>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.depthRGBA_frag = "#include <common>\n#include <logdepthbuf_pars_fragment>\nvec4 pack_depth( const in float depth ) {\n\tconst vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );\n\tconst vec4 bit_mask = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );\n\tvec4 res = mod( depth * bit_shift * vec4( 255 ), vec4( 256 ) ) / vec4( 255 );\n\tres -= res.xxyz * bit_mask;\n\treturn res;\n}\nvoid main() {\n\t#include <logdepthbuf_fragment>\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tgl_FragData[ 0 ] = pack_depth( gl_FragDepthEXT );\n\t#else\n\t\tgl_FragData[ 0 ] = pack_depth( gl_FragCoord.z );\n\t#endif\n}\n",
n.ShaderChunk.depthRGBA_vert = "#include <common>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <skinbase_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.distanceRGBA_frag = "uniform vec3 lightPos;\nvarying vec4 vWorldPosition;\n#include <common>\nvec4 pack1K ( float depth ) {\n\tdepth /= 1000.0;\n\tconst vec4 bitSh = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );\n\tconst vec4 bitMsk = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );\n\tvec4 res = mod( depth * bitSh * vec4( 255 ), vec4( 256 ) ) / vec4( 255 );\n\tres -= res.xxyz * bitMsk;\n\treturn res;\n}\nfloat unpack1K ( vec4 color ) {\n\tconst vec4 bitSh = vec4( 1.0 / ( 256.0 * 256.0 * 256.0 ), 1.0 / ( 256.0 * 256.0 ), 1.0 / 256.0, 1.0 );\n\treturn dot( color, bitSh ) * 1000.0;\n}\nvoid main () {\n\tgl_FragColor = pack1K( length( vWorldPosition.xyz - lightPos.xyz ) );\n}\n",
n.ShaderChunk.distanceRGBA_vert = "varying vec4 vWorldPosition;\n#include <common>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\nvoid main() {\n\t#include <skinbase_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <worldpos_vertex>\n\tvWorldPosition = worldPosition;\n}\n",
n.ShaderChunk.equirect_frag = "uniform sampler2D tEquirect;\nuniform float tFlip;\nvarying vec3 vWorldPosition;\n#include <common>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec3 direction = normalize( vWorldPosition );\n\tvec2 sampleUV;\n\tsampleUV.y = saturate( tFlip * direction.y * -0.5 + 0.5 );\n\tsampleUV.x = atan( direction.z, direction.x ) * RECIPROCAL_PI2 + 0.5;\n\tgl_FragColor = texture2D( tEquirect, sampleUV );\n\t#include <logdepthbuf_fragment>\n}\n",
n.ShaderChunk.equirect_vert = "varying vec3 vWorldPosition;\n#include <common>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\tvWorldPosition = transformDirection( position, modelMatrix );\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.linedashed_frag = "uniform vec3 diffuse;\nuniform float opacity;\nuniform float dashSize;\nuniform float totalSize;\nvarying float vLineDistance;\n#include <common>\n#include <color_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tif ( mod( vLineDistance, totalSize ) > dashSize ) {\n\t\tdiscard;\n\t}\n\tvec3 outgoingLight = vec3( 0.0 );\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <color_fragment>\n\toutgoingLight = diffuseColor.rgb;\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.linedashed_vert = "uniform float scale;\nattribute float lineDistance;\nvarying float vLineDistance;\n#include <common>\n#include <color_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <color_vertex>\n\tvLineDistance = scale * lineDistance;\n\tvec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\n\tgl_Position = projectionMatrix * mvPosition;\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.meshbasic_frag = "uniform vec3 diffuse;\nuniform float opacity;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\n#include <common>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <aomap_pars_fragment>\n#include <envmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\tReflectedLight reflectedLight;\n\treflectedLight.directDiffuse = vec3( 0.0 );\n\treflectedLight.directSpecular = vec3( 0.0 );\n\treflectedLight.indirectDiffuse = diffuseColor.rgb;\n\treflectedLight.indirectSpecular = vec3( 0.0 );\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.indirectDiffuse;\n\t#include <envmap_fragment>\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.meshbasic_vert = "#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <skinbase_vertex>\n\t#ifdef USE_ENVMAP\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#endif\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n}\n",
n.ShaderChunk.meshlambert_frag = "uniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float opacity;\nvarying vec3 vLightFront;\n#ifdef DOUBLE_SIDED\n\tvarying vec3 vLightBack;\n#endif\n#include <common>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <envmap_pars_fragment>\n#include <bsdfs>\n#include <lights_pars>\n#include <fog_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <shadowmask_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\t#include <emissivemap_fragment>\n\treflectedLight.indirectDiffuse = getAmbientLightIrradiance( ambientLightColor );\n\t#include <lightmap_fragment>\n\treflectedLight.indirectDiffuse *= BRDF_Diffuse_Lambert( diffuseColor.rgb );\n\t#ifdef DOUBLE_SIDED\n\t\treflectedLight.directDiffuse = ( gl_FrontFacing ) ? vLightFront : vLightBack;\n\t#else\n\t\treflectedLight.directDiffuse = vLightFront;\n\t#endif\n\treflectedLight.directDiffuse *= BRDF_Diffuse_Lambert( diffuseColor.rgb ) * getShadowMask();\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n\t#include <envmap_fragment>\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.meshlambert_vert = "#define LAMBERT\nvarying vec3 vLightFront;\n#ifdef DOUBLE_SIDED\n\tvarying vec3 vLightBack;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <envmap_pars_vertex>\n#include <bsdfs>\n#include <lights_pars>\n#include <color_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <lights_lambert_vertex>\n\t#include <shadowmap_vertex>\n}\n",
n.ShaderChunk.meshphong_frag = "#define PHONG\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform vec3 specular;\nuniform float shininess;\nuniform float opacity;\n#include <common>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <envmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars>\n#include <lights_phong_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\t#include <normal_fragment>\n\t#include <emissivemap_fragment>\n\t#include <lights_phong_fragment>\n\t#include <lights_template>\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;\n\t#include <envmap_fragment>\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.meshphong_vert = "#define PHONG\nvarying vec3 vViewPosition;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <envmap_pars_vertex>\n#include <lights_phong_pars_vertex>\n#include <color_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n#ifndef FLAT_SHADED\n\tvNormal = normalize( transformedNormal );\n#endif\n\t#include <begin_vertex>\n\t#include <displacementmap_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\tvViewPosition = - mvPosition.xyz;\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <lights_phong_vertex>\n\t#include <shadowmap_vertex>\n}\n",
n.ShaderChunk.meshstandard_frag = "#define STANDARD\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float roughness;\nuniform float metalness;\nuniform float opacity;\nuniform float envMapIntensity;\nvarying vec3 vViewPosition;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\n#include <common>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <envmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <cube_uv_reflection_fragment>\n#include <lights_pars>\n#include <lights_standard_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <roughnessmap_pars_fragment>\n#include <metalnessmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\t#include <roughnessmap_fragment>\n\t#include <metalnessmap_fragment>\n\t#include <normal_fragment>\n\t#include <emissivemap_fragment>\n\t#include <lights_standard_fragment>\n\t#include <lights_template>\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.meshstandard_vert = "#define STANDARD\nvarying vec3 vViewPosition;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n#ifndef FLAT_SHADED\n\tvNormal = normalize( transformedNormal );\n#endif\n\t#include <begin_vertex>\n\t#include <displacementmap_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\tvViewPosition = - mvPosition.xyz;\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <shadowmap_vertex>\n}\n",
n.ShaderChunk.normal_frag = "uniform float opacity;\nvarying vec3 vNormal;\n#include <common>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tgl_FragColor = vec4( 0.5 * normalize( vNormal ) + 0.5, opacity );\n\t#include <logdepthbuf_fragment>\n}\n",
n.ShaderChunk.normal_vert = "varying vec3 vNormal;\n#include <common>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\tvNormal = normalize( normalMatrix * normal );\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n}\n",
n.ShaderChunk.points_frag = "uniform vec3 diffuse;\nuniform float opacity;\n#include <common>\n#include <color_pars_fragment>\n#include <map_particle_pars_fragment>\n#include <fog_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\nvoid main() {\n\tvec3 outgoingLight = vec3( 0.0 );\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_particle_fragment>\n\t#include <color_fragment>\n\t#include <alphatest_fragment>\n\toutgoingLight = diffuseColor.rgb;\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );\n\t#include <premultiplied_alpha_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}\n",
n.ShaderChunk.points_vert = "uniform float size;\nuniform float scale;\n#include <common>\n#include <color_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\nvoid main() {\n\t#include <color_vertex>\n\t#include <begin_vertex>\n\t#include <project_vertex>\n\t#ifdef USE_SIZEATTENUATION\n\t\tgl_PointSize = size * ( scale / - mvPosition.z );\n\t#else\n\t\tgl_PointSize = size;\n\t#endif\n\t#include <logdepthbuf_vertex>\n\t#include <worldpos_vertex>\n\t#include <shadowmap_vertex>\n}\n",
n.ShaderLib = {
    basic: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.common, n.UniformsLib.aomap, n.UniformsLib.fog]),
        vertexShader: n.ShaderChunk.meshbasic_vert,
        fragmentShader: n.ShaderChunk.meshbasic_frag
    },
    lambert: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.common, n.UniformsLib.aomap, n.UniformsLib.lightmap, n.UniformsLib.emissivemap, n.UniformsLib.fog, n.UniformsLib.lights, {
            emissive: {
                type: "c",
                value: new n.Color(0)
            }
        }]),
        vertexShader: n.ShaderChunk.meshlambert_vert,
        fragmentShader: n.ShaderChunk.meshlambert_frag
    },
    phong: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.common, n.UniformsLib.aomap, n.UniformsLib.lightmap, n.UniformsLib.emissivemap, n.UniformsLib.bumpmap, n.UniformsLib.normalmap, n.UniformsLib.displacementmap, n.UniformsLib.fog, n.UniformsLib.lights, {
            emissive: {
                type: "c",
                value: new n.Color(0)
            },
            specular: {
                type: "c",
                value: new n.Color(1118481)
            },
            shininess: {
                type: "f",
                value: 30
            }
        }]),
        vertexShader: n.ShaderChunk.meshphong_vert,
        fragmentShader: n.ShaderChunk.meshphong_frag
    },
    standard: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.common, n.UniformsLib.aomap, n.UniformsLib.lightmap, n.UniformsLib.emissivemap, n.UniformsLib.bumpmap, n.UniformsLib.normalmap, n.UniformsLib.displacementmap, n.UniformsLib.roughnessmap, n.UniformsLib.metalnessmap, n.UniformsLib.fog, n.UniformsLib.lights, {
            emissive: {
                type: "c",
                value: new n.Color(0)
            },
            roughness: {
                type: "f",
                value: .5
            },
            metalness: {
                type: "f",
                value: 0
            },
            envMapIntensity: {
                type: "f",
                value: 1
            }
        }]),
        vertexShader: n.ShaderChunk.meshstandard_vert,
        fragmentShader: n.ShaderChunk.meshstandard_frag
    },
    points: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.points, n.UniformsLib.fog]),
        vertexShader: n.ShaderChunk.points_vert,
        fragmentShader: n.ShaderChunk.points_frag
    },
    dashed: {
        uniforms: n.UniformsUtils.merge([n.UniformsLib.common, n.UniformsLib.fog, {
            scale: {
                type: "f",
                value: 1
            },
            dashSize: {
                type: "f",
                value: 1
            },
            totalSize: {
                type: "f",
                value: 2
            }
        }]),
        vertexShader: n.ShaderChunk.linedashed_vert,
        fragmentShader: n.ShaderChunk.linedashed_frag
    },
    depth: {
        uniforms: {
            mNear: {
                type: "f",
                value: 1
            },
            mFar: {
                type: "f",
                value: 2e3
            },
            opacity: {
                type: "f",
                value: 1
            }
        },
        vertexShader: n.ShaderChunk.depth_vert,
        fragmentShader: n.ShaderChunk.depth_frag
    },
    normal: {
        uniforms: {
            opacity: {
                type: "f",
                value: 1
            }
        },
        vertexShader: n.ShaderChunk.normal_vert,
        fragmentShader: n.ShaderChunk.normal_frag
    },
    cube: {
        uniforms: {
            tCube: {
                type: "t",
                value: null
            },
            tFlip: {
                type: "f",
                value: -1
            }
        },
        vertexShader: n.ShaderChunk.cube_vert,
        fragmentShader: n.ShaderChunk.cube_frag
    },
    equirect: {
        uniforms: {
            tEquirect: {
                type: "t",
                value: null
            },
            tFlip: {
                type: "f",
                value: -1
            }
        },
        vertexShader: n.ShaderChunk.equirect_vert,
        fragmentShader: n.ShaderChunk.equirect_frag
    },
    depthRGBA: {
        uniforms: {},
        vertexShader: n.ShaderChunk.depthRGBA_vert,
        fragmentShader: n.ShaderChunk.depthRGBA_frag
    },
    distanceRGBA: {
        uniforms: {
            lightPos: {
                type: "v3",
                value: new n.Vector3(0,0,0)
            }
        },
        vertexShader: n.ShaderChunk.distanceRGBA_vert,
        fragmentShader: n.ShaderChunk.distanceRGBA_frag
    }
},
n.WebGLRenderer = function(e) {
    function t() {
        return null === ye ? Le : 1
    }
    function i(e, t, i, n) {
        se === !0 && (e *= n,
        t *= n,
        i *= n),
        Ye.clearColor(e, t, i, n)
    }
    function r() {
        Ye.init(),
        Ye.scissor(Ee.copy(Oe).multiplyScalar(Le)),
        Ye.viewport(Te.copy(Fe).multiplyScalar(Le)),
        i(Se.r, Se.g, Se.b, _e)
    }
    function o() {
        Ae = null,
        we = null,
        be = "",
        Ie = -1,
        Ye.reset()
    }
    function a(e) {
        e.preventDefault(),
        o(),
        r(),
        Xe.clear()
    }
    function s(e) {
        var t = e.target;
        t.removeEventListener("dispose", s),
        c(t),
        Ve.textures--
    }
    function l(e) {
        var t = e.target;
        t.removeEventListener("dispose", l),
        u(t),
        Ve.textures--
    }
    function h(e) {
        var t = e.target;
        t.removeEventListener("dispose", h),
        d(t)
    }
    function c(e) {
        var t = Xe.get(e);
        if (e.image && t.__image__webglTextureCube)
            Ge.deleteTexture(t.__image__webglTextureCube);
        else {
            if (void 0 === t.__webglInit)
                return;
            Ge.deleteTexture(t.__webglTexture)
        }
        Xe.delete(e)
    }
    function u(e) {
        var t = Xe.get(e)
          , i = Xe.get(e.texture);
        if (e && void 0 !== i.__webglTexture) {
            if (Ge.deleteTexture(i.__webglTexture),
            e instanceof n.WebGLRenderTargetCube)
                for (var r = 0; r < 6; r++)
                    Ge.deleteFramebuffer(t.__webglFramebuffer[r]),
                    Ge.deleteRenderbuffer(t.__webglDepthbuffer[r]);
            else
                Ge.deleteFramebuffer(t.__webglFramebuffer),
                Ge.deleteRenderbuffer(t.__webglDepthbuffer);
            Xe.delete(e.texture),
            Xe.delete(e)
        }
    }
    function d(e) {
        p(e),
        Xe.delete(e)
    }
    function p(e) {
        var t = Xe.get(e).program;
        e.program = void 0,
        void 0 !== t && Qe.releaseProgram(t)
    }
    function f(e, t, i, r) {
        var o;
        if (i instanceof n.InstancedBufferGeometry && (o = We.get("ANGLE_instanced_arrays"),
        null === o))
            return void console.error("THREE.WebGLRenderer.setupVertexAttributes: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");
        void 0 === r && (r = 0),
        Ye.initAttributes();
        var a = i.attributes
          , s = t.getAttributes()
          , l = e.defaultAttributeValues;
        for (var h in s) {
            var c = s[h];
            if (c >= 0) {
                var u = a[h];
                if (void 0 !== u) {
                    var d = u.itemSize
                      , p = Ze.getAttributeBuffer(u);
                    if (u instanceof n.InterleavedBufferAttribute) {
                        var f = u.data
                          , g = f.stride
                          , m = u.offset;
                        f instanceof n.InstancedInterleavedBuffer ? (Ye.enableAttributeAndDivisor(c, f.meshPerAttribute, o),
                        void 0 === i.maxInstancedCount && (i.maxInstancedCount = f.meshPerAttribute * f.count)) : Ye.enableAttribute(c),
                        Ge.bindBuffer(Ge.ARRAY_BUFFER, p),
                        Ge.vertexAttribPointer(c, d, Ge.FLOAT, !1, g * f.array.BYTES_PER_ELEMENT, (r * g + m) * f.array.BYTES_PER_ELEMENT)
                    } else
                        u instanceof n.InstancedBufferAttribute ? (Ye.enableAttributeAndDivisor(c, u.meshPerAttribute, o),
                        void 0 === i.maxInstancedCount && (i.maxInstancedCount = u.meshPerAttribute * u.count)) : Ye.enableAttribute(c),
                        Ge.bindBuffer(Ge.ARRAY_BUFFER, p),
                        Ge.vertexAttribPointer(c, d, Ge.FLOAT, !1, 0, r * d * 4)
                } else if (void 0 !== l) {
                    var v = l[h];
                    if (void 0 !== v)
                        switch (v.length) {
                        case 2:
                            Ge.vertexAttrib2fv(c, v);
                            break;
                        case 3:
                            Ge.vertexAttrib3fv(c, v);
                            break;
                        case 4:
                            Ge.vertexAttrib4fv(c, v);
                            break;
                        default:
                            Ge.vertexAttrib1fv(c, v)
                        }
                }
            }
        }
        Ye.disableUnusedAttributes()
    }
    function g(e, t) {
        return Math.abs(t[0]) - Math.abs(e[0])
    }
    function m(e, t) {
        return e.object.renderOrder !== t.object.renderOrder ? e.object.renderOrder - t.object.renderOrder : e.material.id !== t.material.id ? e.material.id - t.material.id : e.z !== t.z ? e.z - t.z : e.id - t.id
    }
    function v(e, t) {
        return e.object.renderOrder !== t.object.renderOrder ? e.object.renderOrder - t.object.renderOrder : e.z !== t.z ? t.z - e.z : e.id - t.id
    }
    function A(e, t, i, n, r) {
        var o, a;
        i.transparent ? (o = de,
        a = ++pe) : (o = ce,
        a = ++ue);
        var s = o[a];
        void 0 !== s ? (s.id = e.id,
        s.object = e,
        s.geometry = t,
        s.material = i,
        s.z = ke.z,
        s.group = r) : (s = {
            id: e.id,
            object: e,
            geometry: t,
            material: i,
            z: ke.z,
            group: r
        },
        o.push(s))
    }
    function y(e, t) {
        if (e.visible !== !1) {
            if (e.layers.test(t.layers))
                if (e instanceof n.Light)
                    he.push(e);
                else if (e instanceof n.Sprite)
                    e.frustumCulled !== !1 && Ne.intersectsObject(e) !== !0 || ge.push(e);
                else if (e instanceof n.LensFlare)
                    me.push(e);
                else if (e instanceof n.ImmediateRenderObject)
                    ve.sortObjects === !0 && (ke.setFromMatrixPosition(e.matrixWorld),
                    ke.applyProjection(Be)),
                    A(e, null, e.material, ke.z, null);
                else if ((e instanceof n.Mesh || e instanceof n.Line || e instanceof n.Points) && (e instanceof n.SkinnedMesh && e.skeleton.update(),
                e.frustumCulled === !1 || Ne.intersectsObject(e) === !0)) {
                    var i = e.material;
                    if (i.visible === !0) {
                        ve.sortObjects === !0 && (ke.setFromMatrixPosition(e.matrixWorld),
                        ke.applyProjection(Be));
                        var r = Ze.update(e);
                        if (i instanceof n.MultiMaterial)
                            for (var o = r.groups, a = i.materials, s = 0, l = o.length; s < l; s++) {
                                var h = o[s]
                                  , c = a[h.materialIndex];
                                c.visible === !0 && A(e, r, c, ke.z, h)
                            }
                        else
                            A(e, r, i, ke.z, null)
                    }
                }
            for (var u = e.children, s = 0, l = u.length; s < l; s++)
                y(u[s], t)
        }
    }
    function C(e, t, i, r) {
        for (var o = 0, a = e.length; o < a; o++) {
            var s = e[o]
              , l = s.object
              , h = s.geometry
              , c = void 0 === r ? s.material : r
              , u = s.group;
            if (l.modelViewMatrix.multiplyMatrices(t.matrixWorldInverse, l.matrixWorld),
            l.normalMatrix.getNormalMatrix(l.modelViewMatrix),
            l instanceof n.ImmediateRenderObject) {
                b(c);
                var d = E(t, i, c, l);
                be = "",
                l.render(function(e) {
                    ve.renderBufferImmediate(e, d, c)
                })
            } else
                ve.renderBufferDirect(t, i, h, c, l, u)
        }
    }
    function I(e, t, i) {
        var r = Xe.get(e)
          , o = Qe.getParameters(e, Ue, t, i)
          , a = Qe.getProgramCode(e, o)
          , s = r.program
          , l = !0;
        if (void 0 === s)
            e.addEventListener("dispose", h);
        else if (s.code !== a)
            p(e);
        else {
            if (void 0 !== o.shaderID)
                return;
            l = !1
        }
        if (l) {
            if (o.shaderID) {
                var c = n.ShaderLib[o.shaderID];
                r.__webglShader = {
                    name: e.type,
                    uniforms: n.UniformsUtils.clone(c.uniforms),
                    vertexShader: c.vertexShader,
                    fragmentShader: c.fragmentShader
                }
            } else
                r.__webglShader = {
                    name: e.type,
                    uniforms: e.uniforms,
                    vertexShader: e.vertexShader,
                    fragmentShader: e.fragmentShader
                };
            e.__webglShader = r.__webglShader,
            s = Qe.acquireProgram(e, o, a),
            r.program = s,
            e.program = s
        }
        var u = s.getAttributes();
        if (e.morphTargets) {
            e.numSupportedMorphTargets = 0;
            for (var d = 0; d < ve.maxMorphTargets; d++)
                u["morphTarget" + d] >= 0 && e.numSupportedMorphTargets++
        }
        if (e.morphNormals) {
            e.numSupportedMorphNormals = 0;
            for (var d = 0; d < ve.maxMorphNormals; d++)
                u["morphNormal" + d] >= 0 && e.numSupportedMorphNormals++
        }
        r.uniformsList = [];
        var f = r.__webglShader.uniforms
          , g = r.program.getUniforms();
        for (var m in f) {
            var v = g[m];
            v && r.uniformsList.push([r.__webglShader.uniforms[m], v])
        }
        (e instanceof n.MeshPhongMaterial || e instanceof n.MeshLambertMaterial || e instanceof n.MeshStandardMaterial || e.lights) && (r.lightsHash = Ue.hash,
        f.ambientLightColor.value = Ue.ambient,
        f.directionalLights.value = Ue.directional,
        f.spotLights.value = Ue.spot,
        f.pointLights.value = Ue.point,
        f.hemisphereLights.value = Ue.hemi,
        f.directionalShadowMap.value = Ue.directionalShadowMap,
        f.directionalShadowMatrix.value = Ue.directionalShadowMatrix,
        f.spotShadowMap.value = Ue.spotShadowMap,
        f.spotShadowMatrix.value = Ue.spotShadowMatrix,
        f.pointShadowMap.value = Ue.pointShadowMap,
        f.pointShadowMatrix.value = Ue.pointShadowMatrix),
        r.hasDynamicUniforms = !1;
        for (var A = 0, y = r.uniformsList.length; A < y; A++) {
            var C = r.uniformsList[A][0];
            if (C.dynamic === !0) {
                r.hasDynamicUniforms = !0;
                break
            }
        }
    }
    function b(e) {
        w(e),
        e.transparent === !0 ? Ye.setBlending(e.blending, e.blendEquation, e.blendSrc, e.blendDst, e.blendEquationAlpha, e.blendSrcAlpha, e.blendDstAlpha, e.premultipliedAlpha) : Ye.setBlending(n.NoBlending),
        Ye.setDepthFunc(e.depthFunc),
        Ye.setDepthTest(e.depthTest),
        Ye.setDepthWrite(e.depthWrite),
        Ye.setColorWrite(e.colorWrite),
        Ye.setPolygonOffset(e.polygonOffset, e.polygonOffsetFactor, e.polygonOffsetUnits)
    }
    function w(e) {
        e.side !== n.DoubleSide ? Ye.enable(Ge.CULL_FACE) : Ye.disable(Ge.CULL_FACE),
        Ye.setFlipSided(e.side === n.BackSide)
    }
    function E(e, t, i, r) {
        Me = 0;
        var o = Xe.get(i);
        void 0 === o.program && (i.needsUpdate = !0),
        void 0 !== o.lightsHash && o.lightsHash !== Ue.hash && (i.needsUpdate = !0),
        i.needsUpdate && (I(i, t, r),
        i.needsUpdate = !1);
        var a = !1
          , s = !1
          , l = !1
          , h = o.program
          , c = h.getUniforms()
          , u = o.__webglShader.uniforms;
        if (h.id !== Ae && (Ge.useProgram(h.program),
        Ae = h.id,
        a = !0,
        s = !0,
        l = !0),
        i.id !== Ie && (Ie = i.id,
        s = !0),
        (a || e !== we) && (Ge.uniformMatrix4fv(c.projectionMatrix, !1, e.projectionMatrix.elements),
        je.logarithmicDepthBuffer && Ge.uniform1f(c.logDepthBufFC, 2 / (Math.log(e.far + 1) / Math.LN2)),
        e !== we && (we = e,
        s = !0,
        l = !0),
        (i instanceof n.ShaderMaterial || i instanceof n.MeshPhongMaterial || i instanceof n.MeshStandardMaterial || i.envMap) && void 0 !== c.cameraPosition && (ke.setFromMatrixPosition(e.matrixWorld),
        Ge.uniform3f(c.cameraPosition, ke.x, ke.y, ke.z)),
        (i instanceof n.MeshPhongMaterial || i instanceof n.MeshLambertMaterial || i instanceof n.MeshBasicMaterial || i instanceof n.MeshStandardMaterial || i instanceof n.ShaderMaterial || i.skinning) && void 0 !== c.viewMatrix && Ge.uniformMatrix4fv(c.viewMatrix, !1, e.matrixWorldInverse.elements),
        void 0 !== c.toneMappingExposure && Ge.uniform1f(c.toneMappingExposure, ve.toneMappingExposure),
        void 0 !== c.toneMappingWhitePoint && Ge.uniform1f(c.toneMappingWhitePoint, ve.toneMappingWhitePoint)),
        i.skinning)
            if (r.bindMatrix && void 0 !== c.bindMatrix && Ge.uniformMatrix4fv(c.bindMatrix, !1, r.bindMatrix.elements),
            r.bindMatrixInverse && void 0 !== c.bindMatrixInverse && Ge.uniformMatrix4fv(c.bindMatrixInverse, !1, r.bindMatrixInverse.elements),
            je.floatVertexTextures && r.skeleton && r.skeleton.useVertexTexture) {
                if (void 0 !== c.boneTexture) {
                    var d = N();
                    Ge.uniform1i(c.boneTexture, d),
                    ve.setTexture(r.skeleton.boneTexture, d)
                }
                void 0 !== c.boneTextureWidth && Ge.uniform1i(c.boneTextureWidth, r.skeleton.boneTextureWidth),
                void 0 !== c.boneTextureHeight && Ge.uniform1i(c.boneTextureHeight, r.skeleton.boneTextureHeight)
            } else
                r.skeleton && r.skeleton.boneMatrices && void 0 !== c.boneGlobalMatrices && Ge.uniformMatrix4fv(c.boneGlobalMatrices, !1, r.skeleton.boneMatrices);
        return s && ((i instanceof n.MeshPhongMaterial || i instanceof n.MeshLambertMaterial || i instanceof n.MeshStandardMaterial || i.lights) && D(u, l),
        t && i.fog && P(u, t),
        (i instanceof n.MeshBasicMaterial || i instanceof n.MeshLambertMaterial || i instanceof n.MeshPhongMaterial || i instanceof n.MeshStandardMaterial) && T(u, i),
        i instanceof n.LineBasicMaterial ? M(u, i) : i instanceof n.LineDashedMaterial ? (M(u, i),
        S(u, i)) : i instanceof n.PointsMaterial ? _(u, i) : i instanceof n.MeshLambertMaterial ? R(u, i) : i instanceof n.MeshPhongMaterial ? L(u, i) : i instanceof n.MeshStandardMaterial ? O(u, i) : i instanceof n.MeshDepthMaterial ? (u.mNear.value = e.near,
        u.mFar.value = e.far,
        u.opacity.value = i.opacity) : i instanceof n.MeshNormalMaterial && (u.opacity.value = i.opacity),
        k(o.uniformsList)),
        F(c, r),
        void 0 !== c.modelMatrix && Ge.uniformMatrix4fv(c.modelMatrix, !1, r.matrixWorld.elements),
        o.hasDynamicUniforms === !0 && x(o.uniformsList, r, e),
        h
    }
    function x(e, t, i) {
        for (var n = [], r = 0, o = e.length; r < o; r++) {
            var a = e[r][0]
              , s = a.onUpdateCallback;
            void 0 !== s && (s.bind(a)(t, i),
            n.push(e[r]))
        }
        k(n)
    }
    function T(e, t) {
        e.opacity.value = t.opacity,
        e.diffuse.value = t.color,
        t.emissive && e.emissive.value.copy(t.emissive).multiplyScalar(t.emissiveIntensity),
        e.map.value = t.map,
        e.specularMap.value = t.specularMap,
        e.alphaMap.value = t.alphaMap,
        t.aoMap && (e.aoMap.value = t.aoMap,
        e.aoMapIntensity.value = t.aoMapIntensity);
        var i;
        if (t.map ? i = t.map : t.specularMap ? i = t.specularMap : t.displacementMap ? i = t.displacementMap : t.normalMap ? i = t.normalMap : t.bumpMap ? i = t.bumpMap : t.roughnessMap ? i = t.roughnessMap : t.metalnessMap ? i = t.metalnessMap : t.alphaMap ? i = t.alphaMap : t.emissiveMap && (i = t.emissiveMap),
        void 0 !== i) {
            i instanceof n.WebGLRenderTarget && (i = i.texture);
            var r = i.offset
              , o = i.repeat;
            e.offsetRepeat.value.set(r.x, r.y, o.x, o.y)
        }
        e.envMap.value = t.envMap,
        e.flipEnvMap.value = t.envMap instanceof n.WebGLRenderTargetCube ? 1 : -1,
        e.reflectivity.value = t.reflectivity,
        e.refractionRatio.value = t.refractionRatio
    }
    function M(e, t) {
        e.diffuse.value = t.color,
        e.opacity.value = t.opacity
    }
    function S(e, t) {
        e.dashSize.value = t.dashSize,
        e.totalSize.value = t.dashSize + t.gapSize,
        e.scale.value = t.scale
    }
    function _(e, t) {
        if (e.diffuse.value = t.color,
        e.opacity.value = t.opacity,
        e.size.value = t.size * Le,
        e.scale.value = te.clientHeight / 2,
        e.map.value = t.map,
        null !== t.map) {
            var i = t.map.offset
              , n = t.map.repeat;
            e.offsetRepeat.value.set(i.x, i.y, n.x, n.y)
        }
    }
    function P(e, t) {
        e.fogColor.value = t.color,
        t instanceof n.Fog ? (e.fogNear.value = t.near,
        e.fogFar.value = t.far) : t instanceof n.FogExp2 && (e.fogDensity.value = t.density)
    }
    function R(e, t) {
        t.lightMap && (e.lightMap.value = t.lightMap,
        e.lightMapIntensity.value = t.lightMapIntensity),
        t.emissiveMap && (e.emissiveMap.value = t.emissiveMap)
    }
    function L(e, t) {
        e.specular.value = t.specular,
        e.shininess.value = Math.max(t.shininess, 1e-4),
        t.lightMap && (e.lightMap.value = t.lightMap,
        e.lightMapIntensity.value = t.lightMapIntensity),
        t.emissiveMap && (e.emissiveMap.value = t.emissiveMap),
        t.bumpMap && (e.bumpMap.value = t.bumpMap,
        e.bumpScale.value = t.bumpScale),
        t.normalMap && (e.normalMap.value = t.normalMap,
        e.normalScale.value.copy(t.normalScale)),
        t.displacementMap && (e.displacementMap.value = t.displacementMap,
        e.displacementScale.value = t.displacementScale,
        e.displacementBias.value = t.displacementBias)
    }
    function O(e, t) {
        e.roughness.value = t.roughness,
        e.metalness.value = t.metalness,
        t.roughnessMap && (e.roughnessMap.value = t.roughnessMap),
        t.metalnessMap && (e.metalnessMap.value = t.metalnessMap),
        t.lightMap && (e.lightMap.value = t.lightMap,
        e.lightMapIntensity.value = t.lightMapIntensity),
        t.emissiveMap && (e.emissiveMap.value = t.emissiveMap),
        t.bumpMap && (e.bumpMap.value = t.bumpMap,
        e.bumpScale.value = t.bumpScale),
        t.normalMap && (e.normalMap.value = t.normalMap,
        e.normalScale.value.copy(t.normalScale)),
        t.displacementMap && (e.displacementMap.value = t.displacementMap,
        e.displacementScale.value = t.displacementScale,
        e.displacementBias.value = t.displacementBias),
        t.envMap && (e.envMapIntensity.value = t.envMapIntensity)
    }
    function D(e, t) {
        e.ambientLightColor.needsUpdate = t,
        e.directionalLights.needsUpdate = t,
        e.pointLights.needsUpdate = t,
        e.spotLights.needsUpdate = t,
        e.hemisphereLights.needsUpdate = t
    }
    function F(e, t) {
        Ge.uniformMatrix4fv(e.modelViewMatrix, !1, t.modelViewMatrix.elements),
        e.normalMatrix && Ge.uniformMatrix3fv(e.normalMatrix, !1, t.normalMatrix.elements)
    }
    function N() {
        var e = Me;
        return e >= je.maxTextures && console.warn("WebGLRenderer: trying to use " + e + " texture units while this GPU supports only " + je.maxTextures),
        Me += 1,
        e
    }
    function B(e, t, i, r) {
        var o, a;
        if ("1i" === t)
            Ge.uniform1i(i, r);
        else if ("1f" === t)
            Ge.uniform1f(i, r);
        else if ("2f" === t)
            Ge.uniform2f(i, r[0], r[1]);
        else if ("3f" === t)
            Ge.uniform3f(i, r[0], r[1], r[2]);
        else if ("4f" === t)
            Ge.uniform4f(i, r[0], r[1], r[2], r[3]);
        else if ("1iv" === t)
            Ge.uniform1iv(i, r);
        else if ("3iv" === t)
            Ge.uniform3iv(i, r);
        else if ("1fv" === t)
            Ge.uniform1fv(i, r);
        else if ("2fv" === t)
            Ge.uniform2fv(i, r);
        else if ("3fv" === t)
            Ge.uniform3fv(i, r);
        else if ("4fv" === t)
            Ge.uniform4fv(i, r);
        else if ("Matrix2fv" === t)
            Ge.uniformMatrix2fv(i, !1, r);
        else if ("Matrix3fv" === t)
            Ge.uniformMatrix3fv(i, !1, r);
        else if ("Matrix4fv" === t)
            Ge.uniformMatrix4fv(i, !1, r);
        else if ("i" === t)
            Ge.uniform1i(i, r);
        else if ("f" === t)
            Ge.uniform1f(i, r);
        else if ("v2" === t)
            Ge.uniform2f(i, r.x, r.y);
        else if ("v3" === t)
            Ge.uniform3f(i, r.x, r.y, r.z);
        else if ("v4" === t)
            Ge.uniform4f(i, r.x, r.y, r.z, r.w);
        else if ("c" === t)
            Ge.uniform3f(i, r.r, r.g, r.b);
        else if ("s" === t) {
            var s = e.properties;
            for (var l in s) {
                var h = s[l]
                  , c = i[l]
                  , u = r[l];
                B(h, h.type, c, u)
            }
        } else if ("sa" === t)
            for (var s = e.properties, d = 0, p = r.length; d < p; d++)
                for (var l in s) {
                    var h = s[l]
                      , c = i[d][l]
                      , u = r[d][l];
                    B(h, h.type, c, u)
                }
        else if ("iv1" === t)
            Ge.uniform1iv(i, r);
        else if ("iv" === t)
            Ge.uniform3iv(i, r);
        else if ("fv1" === t)
            Ge.uniform1fv(i, r);
        else if ("fv" === t)
            Ge.uniform3fv(i, r);
        else if ("v2v" === t) {
            void 0 === e._array && (e._array = new Float32Array(2 * r.length));
            for (var d = 0, f = 0, g = r.length; d < g; d++,
            f += 2)
                e._array[f + 0] = r[d].x,
                e._array[f + 1] = r[d].y;
            Ge.uniform2fv(i, e._array)
        } else if ("v3v" === t) {
            void 0 === e._array && (e._array = new Float32Array(3 * r.length));
            for (var d = 0, m = 0, g = r.length; d < g; d++,
            m += 3)
                e._array[m + 0] = r[d].x,
                e._array[m + 1] = r[d].y,
                e._array[m + 2] = r[d].z;
            Ge.uniform3fv(i, e._array)
        } else if ("v4v" === t) {
            void 0 === e._array && (e._array = new Float32Array(4 * r.length));
            for (var d = 0, v = 0, g = r.length; d < g; d++,
            v += 4)
                e._array[v + 0] = r[d].x,
                e._array[v + 1] = r[d].y,
                e._array[v + 2] = r[d].z,
                e._array[v + 3] = r[d].w;
            Ge.uniform4fv(i, e._array)
        } else if ("m2" === t)
            Ge.uniformMatrix2fv(i, !1, r.elements);
        else if ("m3" === t)
            Ge.uniformMatrix3fv(i, !1, r.elements);
        else if ("m3v" === t) {
            void 0 === e._array && (e._array = new Float32Array(9 * r.length));
            for (var d = 0, g = r.length; d < g; d++)
                r[d].flattenToArrayOffset(e._array, 9 * d);
            Ge.uniformMatrix3fv(i, !1, e._array)
        } else if ("m4" === t)
            Ge.uniformMatrix4fv(i, !1, r.elements);
        else if ("m4v" === t) {
            void 0 === e._array && (e._array = new Float32Array(16 * r.length));
            for (var d = 0, g = r.length; d < g; d++)
                r[d].flattenToArrayOffset(e._array, 16 * d);
            Ge.uniformMatrix4fv(i, !1, e._array)
        } else if ("t" === t) {
            if (o = r,
            a = N(),
            Ge.uniform1i(i, a),
            !o)
                return;
            o instanceof n.CubeTexture || Array.isArray(o.image) && 6 === o.image.length ? Y(o, a) : o instanceof n.WebGLRenderTargetCube ? X(o.texture, a) : o instanceof n.WebGLRenderTarget ? ve.setTexture(o.texture, a) : ve.setTexture(o, a)
        } else if ("tv" === t) {
            void 0 === e._array && (e._array = []);
            for (var d = 0, g = e.value.length; d < g; d++)
                e._array[d] = N();
            Ge.uniform1iv(i, e._array);
            for (var d = 0, g = e.value.length; d < g; d++)
                o = e.value[d],
                a = e._array[d],
                o && (o instanceof n.CubeTexture || o.image instanceof Array && 6 === o.image.length ? Y(o, a) : o instanceof n.WebGLRenderTarget ? ve.setTexture(o.texture, a) : o instanceof n.WebGLRenderTargetCube ? X(o.texture, a) : ve.setTexture(o, a))
        } else
            console.warn("THREE.WebGLRenderer: Unknown uniform type: " + t)
    }
    function k(e) {
        for (var t = 0, i = e.length; t < i; t++) {
            var n = e[t][0];
            if (n.needsUpdate !== !1) {
                var r = n.type
                  , o = e[t][1]
                  , a = n.value;
                B(n, r, o, a)
            }
        }
    }
    function U(e, t) {
        var i, r, o, a, s, l, h = 0, c = 0, u = 0, d = t.matrixWorldInverse, p = 0, f = 0, g = 0, m = 0, v = 0;
        for (Ue.shadowsPointLight = 0,
        i = 0,
        r = e.length; i < r; i++)
            if (o = e[i],
            a = o.color,
            s = o.intensity,
            l = o.distance,
            o instanceof n.AmbientLight)
                h += a.r * s,
                c += a.g * s,
                u += a.b * s;
            else if (o instanceof n.DirectionalLight) {
                var A = qe.get(o);
                A.color.copy(o.color).multiplyScalar(o.intensity),
                A.direction.setFromMatrixPosition(o.matrixWorld),
                ke.setFromMatrixPosition(o.target.matrixWorld),
                A.direction.sub(ke),
                A.direction.transformDirection(d),
                A.shadow = o.castShadow,
                o.castShadow && (A.shadowBias = o.shadow.bias,
                A.shadowRadius = o.shadow.radius,
                A.shadowMapSize = o.shadow.mapSize,
                Ue.shadows[v++] = o),
                Ue.directionalShadowMap[p] = o.shadow.map,
                Ue.directionalShadowMatrix[p] = o.shadow.matrix,
                Ue.directional[p++] = A
            } else if (o instanceof n.SpotLight) {
                var A = qe.get(o);
                A.position.setFromMatrixPosition(o.matrixWorld),
                A.position.applyMatrix4(d),
                A.color.copy(a).multiplyScalar(s),
                A.distance = l,
                A.direction.setFromMatrixPosition(o.matrixWorld),
                ke.setFromMatrixPosition(o.target.matrixWorld),
                A.direction.sub(ke),
                A.direction.transformDirection(d),
                A.coneCos = Math.cos(o.angle),
                A.penumbraCos = Math.cos(o.angle * (1 - o.penumbra)),
                A.decay = 0 === o.distance ? 0 : o.decay,
                A.shadow = o.castShadow,
                o.castShadow && (A.shadowBias = o.shadow.bias,
                A.shadowRadius = o.shadow.radius,
                A.shadowMapSize = o.shadow.mapSize,
                Ue.shadows[v++] = o),
                Ue.spotShadowMap[g] = o.shadow.map,
                Ue.spotShadowMatrix[g] = o.shadow.matrix,
                Ue.spot[g++] = A
            } else if (o instanceof n.PointLight) {
                var A = qe.get(o);
                A.position.setFromMatrixPosition(o.matrixWorld),
                A.position.applyMatrix4(d),
                A.color.copy(o.color).multiplyScalar(o.intensity),
                A.distance = o.distance,
                A.decay = 0 === o.distance ? 0 : o.decay,
                A.shadow = o.castShadow,
                o.castShadow && (A.shadowBias = o.shadow.bias,
                A.shadowRadius = o.shadow.radius,
                A.shadowMapSize = o.shadow.mapSize,
                Ue.shadows[v++] = o),
                Ue.pointShadowMap[f] = o.shadow.map,
                void 0 === Ue.pointShadowMatrix[f] && (Ue.pointShadowMatrix[f] = new n.Matrix4),
                ke.setFromMatrixPosition(o.matrixWorld).negate(),
                Ue.pointShadowMatrix[f].identity().setPosition(ke),
                Ue.point[f++] = A
            } else if (o instanceof n.HemisphereLight) {
                var A = qe.get(o);
                A.direction.setFromMatrixPosition(o.matrixWorld),
                A.direction.transformDirection(d),
                A.direction.normalize(),
                A.skyColor.copy(o.color).multiplyScalar(s),
                A.groundColor.copy(o.groundColor).multiplyScalar(s),
                Ue.hemi[m++] = A
            }
        Ue.ambient[0] = h,
        Ue.ambient[1] = c,
        Ue.ambient[2] = u,
        Ue.directional.length = p,
        Ue.spot.length = g,
        Ue.point.length = f,
        Ue.hemi.length = m,
        Ue.shadows.length = v,
        Ue.hash = p + "," + f + "," + g + "," + m + "," + v
    }
    function V(e, t, i) {
        var r;
        if (i ? (Ge.texParameteri(e, Ge.TEXTURE_WRAP_S, ee(t.wrapS)),
        Ge.texParameteri(e, Ge.TEXTURE_WRAP_T, ee(t.wrapT)),
        Ge.texParameteri(e, Ge.TEXTURE_MAG_FILTER, ee(t.magFilter)),
        Ge.texParameteri(e, Ge.TEXTURE_MIN_FILTER, ee(t.minFilter))) : (Ge.texParameteri(e, Ge.TEXTURE_WRAP_S, Ge.CLAMP_TO_EDGE),
        Ge.texParameteri(e, Ge.TEXTURE_WRAP_T, Ge.CLAMP_TO_EDGE),
        t.wrapS === n.ClampToEdgeWrapping && t.wrapT === n.ClampToEdgeWrapping || console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.wrapS and Texture.wrapT should be set to THREE.ClampToEdgeWrapping.", t),
        Ge.texParameteri(e, Ge.TEXTURE_MAG_FILTER, $(t.magFilter)),
        Ge.texParameteri(e, Ge.TEXTURE_MIN_FILTER, $(t.minFilter)),
        t.minFilter !== n.NearestFilter && t.minFilter !== n.LinearFilter && console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.minFilter should be set to THREE.NearestFilter or THREE.LinearFilter.", t)),
        r = We.get("EXT_texture_filter_anisotropic")) {
            if (t.type === n.FloatType && null === We.get("OES_texture_float_linear"))
                return;
            if (t.type === n.HalfFloatType && null === We.get("OES_texture_half_float_linear"))
                return;
            (t.anisotropy > 1 || Xe.get(t).__currentAnisotropy) && (Ge.texParameterf(e, r.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(t.anisotropy, ve.getMaxAnisotropy())),
            Xe.get(t).__currentAnisotropy = t.anisotropy)
        }
    }
    function z(e, t, i) {
        void 0 === e.__webglInit && (e.__webglInit = !0,
        t.addEventListener("dispose", s),
        e.__webglTexture = Ge.createTexture(),
        Ve.textures++),
        Ye.activeTexture(Ge.TEXTURE0 + i),
        Ye.bindTexture(Ge.TEXTURE_2D, e.__webglTexture),
        Ge.pixelStorei(Ge.UNPACK_FLIP_Y_WEBGL, t.flipY),
        Ge.pixelStorei(Ge.UNPACK_PREMULTIPLY_ALPHA_WEBGL, t.premultiplyAlpha),
        Ge.pixelStorei(Ge.UNPACK_ALIGNMENT, t.unpackAlignment);
        var r = G(t.image, je.maxTextureSize);
        W(t) && H(r) === !1 && (r = j(r));
        var o = H(r)
          , a = ee(t.format)
          , l = ee(t.type);
        V(Ge.TEXTURE_2D, t, o);
        var h, c = t.mipmaps;
        if (t instanceof n.DataTexture)
            if (c.length > 0 && o) {
                for (var u = 0, d = c.length; u < d; u++)
                    h = c[u],
                    Ye.texImage2D(Ge.TEXTURE_2D, u, a, h.width, h.height, 0, a, l, h.data);
                t.generateMipmaps = !1
            } else
                Ye.texImage2D(Ge.TEXTURE_2D, 0, a, r.width, r.height, 0, a, l, r.data);
        else if (t instanceof n.CompressedTexture)
            for (var u = 0, d = c.length; u < d; u++)
                h = c[u],
                t.format !== n.RGBAFormat && t.format !== n.RGBFormat ? Ye.getCompressedTextureFormats().indexOf(a) > -1 ? Ye.compressedTexImage2D(Ge.TEXTURE_2D, u, a, h.width, h.height, 0, h.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()") : Ye.texImage2D(Ge.TEXTURE_2D, u, a, h.width, h.height, 0, a, l, h.data);
        else if (c.length > 0 && o) {
            for (var u = 0, d = c.length; u < d; u++)
                h = c[u],
                Ye.texImage2D(Ge.TEXTURE_2D, u, a, a, l, h);
            t.generateMipmaps = !1
        } else
            Ye.texImage2D(Ge.TEXTURE_2D, 0, a, a, l, r);
        t.generateMipmaps && o && Ge.generateMipmap(Ge.TEXTURE_2D),
        e.__version = t.version,
        t.onUpdate && t.onUpdate(t)
    }
    function G(e, t) {
        if (e.width > t || e.height > t) {
            var i = t / Math.max(e.width, e.height)
              , n = document.createElement("canvas");
            n.width = Math.floor(e.width * i),
            n.height = Math.floor(e.height * i);
            var r = n.getContext("2d");
            return r.drawImage(e, 0, 0, e.width, e.height, 0, 0, n.width, n.height),
            console.warn("THREE.WebGLRenderer: image is too big (" + e.width + "x" + e.height + "). Resized to " + n.width + "x" + n.height, e),
            n
        }
        return e
    }
    function H(e) {
        return n.Math.isPowerOfTwo(e.width) && n.Math.isPowerOfTwo(e.height)
    }
    function W(e) {
        return e.wrapS !== n.ClampToEdgeWrapping || e.wrapT !== n.ClampToEdgeWrapping || e.minFilter !== n.NearestFilter && e.minFilter !== n.LinearFilter
    }
    function j(e) {
        if (e instanceof HTMLImageElement || e instanceof HTMLCanvasElement) {
            var t = document.createElement("canvas");
            t.width = n.Math.nearestPowerOfTwo(e.width),
            t.height = n.Math.nearestPowerOfTwo(e.height);
            var i = t.getContext("2d");
            return i.drawImage(e, 0, 0, t.width, t.height),
            console.warn("THREE.WebGLRenderer: image is not power of two (" + e.width + "x" + e.height + "). Resized to " + t.width + "x" + t.height, e),
            t
        }
        return e
    }
    function Y(e, t) {
        var i = Xe.get(e);
        if (6 === e.image.length)
            if (e.version > 0 && i.__version !== e.version) {
                i.__image__webglTextureCube || (e.addEventListener("dispose", s),
                i.__image__webglTextureCube = Ge.createTexture(),
                Ve.textures++),
                Ye.activeTexture(Ge.TEXTURE0 + t),
                Ye.bindTexture(Ge.TEXTURE_CUBE_MAP, i.__image__webglTextureCube),
                Ge.pixelStorei(Ge.UNPACK_FLIP_Y_WEBGL, e.flipY);
                for (var r = e instanceof n.CompressedTexture, o = e.image[0]instanceof n.DataTexture, a = [], l = 0; l < 6; l++)
                    !ve.autoScaleCubemaps || r || o ? a[l] = o ? e.image[l].image : e.image[l] : a[l] = G(e.image[l], je.maxCubemapSize);
                var h = a[0]
                  , c = H(h)
                  , u = ee(e.format)
                  , d = ee(e.type);
                V(Ge.TEXTURE_CUBE_MAP, e, c);
                for (var l = 0; l < 6; l++)
                    if (r)
                        for (var p, f = a[l].mipmaps, g = 0, m = f.length; g < m; g++)
                            p = f[g],
                            e.format !== n.RGBAFormat && e.format !== n.RGBFormat ? Ye.getCompressedTextureFormats().indexOf(u) > -1 ? Ye.compressedTexImage2D(Ge.TEXTURE_CUBE_MAP_POSITIVE_X + l, g, u, p.width, p.height, 0, p.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setCubeTexture()") : Ye.texImage2D(Ge.TEXTURE_CUBE_MAP_POSITIVE_X + l, g, u, p.width, p.height, 0, u, d, p.data);
                    else
                        o ? Ye.texImage2D(Ge.TEXTURE_CUBE_MAP_POSITIVE_X + l, 0, u, a[l].width, a[l].height, 0, u, d, a[l].data) : Ye.texImage2D(Ge.TEXTURE_CUBE_MAP_POSITIVE_X + l, 0, u, u, d, a[l]);
                e.generateMipmaps && c && Ge.generateMipmap(Ge.TEXTURE_CUBE_MAP),
                i.__version = e.version,
                e.onUpdate && e.onUpdate(e)
            } else
                Ye.activeTexture(Ge.TEXTURE0 + t),
                Ye.bindTexture(Ge.TEXTURE_CUBE_MAP, i.__image__webglTextureCube)
    }
    function X(e, t) {
        Ye.activeTexture(Ge.TEXTURE0 + t),
        Ye.bindTexture(Ge.TEXTURE_CUBE_MAP, Xe.get(e).__webglTexture)
    }
    function Z(e, t, i, n) {
        var r = ee(t.texture.format)
          , o = ee(t.texture.type);
        Ye.texImage2D(n, 0, r, t.width, t.height, 0, r, o, null),
        Ge.bindFramebuffer(Ge.FRAMEBUFFER, e),
        Ge.framebufferTexture2D(Ge.FRAMEBUFFER, i, n, Xe.get(t.texture).__webglTexture, 0),
        Ge.bindFramebuffer(Ge.FRAMEBUFFER, null)
    }
    function Q(e, t) {
        Ge.bindRenderbuffer(Ge.RENDERBUFFER, e),
        t.depthBuffer && !t.stencilBuffer ? (Ge.renderbufferStorage(Ge.RENDERBUFFER, Ge.DEPTH_COMPONENT16, t.width, t.height),
        Ge.framebufferRenderbuffer(Ge.FRAMEBUFFER, Ge.DEPTH_ATTACHMENT, Ge.RENDERBUFFER, e)) : t.depthBuffer && t.stencilBuffer ? (Ge.renderbufferStorage(Ge.RENDERBUFFER, Ge.DEPTH_STENCIL, t.width, t.height),
        Ge.framebufferRenderbuffer(Ge.FRAMEBUFFER, Ge.DEPTH_STENCIL_ATTACHMENT, Ge.RENDERBUFFER, e)) : Ge.renderbufferStorage(Ge.RENDERBUFFER, Ge.RGBA4, t.width, t.height),
        Ge.bindRenderbuffer(Ge.RENDERBUFFER, null)
    }
    function q(e) {
        var t = Xe.get(e)
          , i = e instanceof n.WebGLRenderTargetCube;
        if (i) {
            t.__webglDepthbuffer = [];
            for (var r = 0; r < 6; r++)
                Ge.bindFramebuffer(Ge.FRAMEBUFFER, t.__webglFramebuffer[r]),
                t.__webglDepthbuffer[r] = Ge.createRenderbuffer(),
                Q(t.__webglDepthbuffer[r], e)
        } else
            Ge.bindFramebuffer(Ge.FRAMEBUFFER, t.__webglFramebuffer),
            t.__webglDepthbuffer = Ge.createRenderbuffer(),
            Q(t.__webglDepthbuffer, e);
        Ge.bindFramebuffer(Ge.FRAMEBUFFER, null)
    }
    function K(e) {
        var t = Xe.get(e)
          , i = Xe.get(e.texture);
        e.addEventListener("dispose", l),
        i.__webglTexture = Ge.createTexture(),
        Ve.textures++;
        var r = e instanceof n.WebGLRenderTargetCube
          , o = n.Math.isPowerOfTwo(e.width) && n.Math.isPowerOfTwo(e.height);
        if (r) {
            t.__webglFramebuffer = [];
            for (var a = 0; a < 6; a++)
                t.__webglFramebuffer[a] = Ge.createFramebuffer()
        } else
            t.__webglFramebuffer = Ge.createFramebuffer();
        if (r) {
            Ye.bindTexture(Ge.TEXTURE_CUBE_MAP, i.__webglTexture),
            V(Ge.TEXTURE_CUBE_MAP, e.texture, o);
            for (var a = 0; a < 6; a++)
                Z(t.__webglFramebuffer[a], e, Ge.COLOR_ATTACHMENT0, Ge.TEXTURE_CUBE_MAP_POSITIVE_X + a);
            e.texture.generateMipmaps && o && Ge.generateMipmap(Ge.TEXTURE_CUBE_MAP),
            Ye.bindTexture(Ge.TEXTURE_CUBE_MAP, null)
        } else
            Ye.bindTexture(Ge.TEXTURE_2D, i.__webglTexture),
            V(Ge.TEXTURE_2D, e.texture, o),
            Z(t.__webglFramebuffer, e, Ge.COLOR_ATTACHMENT0, Ge.TEXTURE_2D),
            e.texture.generateMipmaps && o && Ge.generateMipmap(Ge.TEXTURE_2D),
            Ye.bindTexture(Ge.TEXTURE_2D, null);
        e.depthBuffer && q(e)
    }
    function J(e) {
        var t = e instanceof n.WebGLRenderTargetCube ? Ge.TEXTURE_CUBE_MAP : Ge.TEXTURE_2D
          , i = Xe.get(e.texture).__webglTexture;
        Ye.bindTexture(t, i),
        Ge.generateMipmap(t),
        Ye.bindTexture(t, null)
    }
    function $(e) {
        return e === n.NearestFilter || e === n.NearestMipMapNearestFilter || e === n.NearestMipMapLinearFilter ? Ge.NEAREST : Ge.LINEAR
    }
    function ee(e) {
        var t;
        if (e === n.RepeatWrapping)
            return Ge.REPEAT;
        if (e === n.ClampToEdgeWrapping)
            return Ge.CLAMP_TO_EDGE;
        if (e === n.MirroredRepeatWrapping)
            return Ge.MIRRORED_REPEAT;
        if (e === n.NearestFilter)
            return Ge.NEAREST;
        if (e === n.NearestMipMapNearestFilter)
            return Ge.NEAREST_MIPMAP_NEAREST;
        if (e === n.NearestMipMapLinearFilter)
            return Ge.NEAREST_MIPMAP_LINEAR;
        if (e === n.LinearFilter)
            return Ge.LINEAR;
        if (e === n.LinearMipMapNearestFilter)
            return Ge.LINEAR_MIPMAP_NEAREST;
        if (e === n.LinearMipMapLinearFilter)
            return Ge.LINEAR_MIPMAP_LINEAR;
        if (e === n.UnsignedByteType)
            return Ge.UNSIGNED_BYTE;
        if (e === n.UnsignedShort4444Type)
            return Ge.UNSIGNED_SHORT_4_4_4_4;
        if (e === n.UnsignedShort5551Type)
            return Ge.UNSIGNED_SHORT_5_5_5_1;
        if (e === n.UnsignedShort565Type)
            return Ge.UNSIGNED_SHORT_5_6_5;
        if (e === n.ByteType)
            return Ge.BYTE;
        if (e === n.ShortType)
            return Ge.SHORT;
        if (e === n.UnsignedShortType)
            return Ge.UNSIGNED_SHORT;
        if (e === n.IntType)
            return Ge.INT;
        if (e === n.UnsignedIntType)
            return Ge.UNSIGNED_INT;
        if (e === n.FloatType)
            return Ge.FLOAT;
        if (t = We.get("OES_texture_half_float"),
        null !== t && e === n.HalfFloatType)
            return t.HALF_FLOAT_OES;
        if (e === n.AlphaFormat)
            return Ge.ALPHA;
        if (e === n.RGBFormat)
            return Ge.RGB;
        if (e === n.RGBAFormat)
            return Ge.RGBA;
        if (e === n.LuminanceFormat)
            return Ge.LUMINANCE;
        if (e === n.LuminanceAlphaFormat)
            return Ge.LUMINANCE_ALPHA;
        if (e === n.AddEquation)
            return Ge.FUNC_ADD;
        if (e === n.SubtractEquation)
            return Ge.FUNC_SUBTRACT;
        if (e === n.ReverseSubtractEquation)
            return Ge.FUNC_REVERSE_SUBTRACT;
        if (e === n.ZeroFactor)
            return Ge.ZERO;
        if (e === n.OneFactor)
            return Ge.ONE;
        if (e === n.SrcColorFactor)
            return Ge.SRC_COLOR;
        if (e === n.OneMinusSrcColorFactor)
            return Ge.ONE_MINUS_SRC_COLOR;
        if (e === n.SrcAlphaFactor)
            return Ge.SRC_ALPHA;
        if (e === n.OneMinusSrcAlphaFactor)
            return Ge.ONE_MINUS_SRC_ALPHA;
        if (e === n.DstAlphaFactor)
            return Ge.DST_ALPHA;
        if (e === n.OneMinusDstAlphaFactor)
            return Ge.ONE_MINUS_DST_ALPHA;
        if (e === n.DstColorFactor)
            return Ge.DST_COLOR;
        if (e === n.OneMinusDstColorFactor)
            return Ge.ONE_MINUS_DST_COLOR;
        if (e === n.SrcAlphaSaturateFactor)
            return Ge.SRC_ALPHA_SATURATE;
        if (t = We.get("WEBGL_compressed_texture_s3tc"),
        null !== t) {
            if (e === n.RGB_S3TC_DXT1_Format)
                return t.COMPRESSED_RGB_S3TC_DXT1_EXT;
            if (e === n.RGBA_S3TC_DXT1_Format)
                return t.COMPRESSED_RGBA_S3TC_DXT1_EXT;
            if (e === n.RGBA_S3TC_DXT3_Format)
                return t.COMPRESSED_RGBA_S3TC_DXT3_EXT;
            if (e === n.RGBA_S3TC_DXT5_Format)
                return t.COMPRESSED_RGBA_S3TC_DXT5_EXT
        }
        if (t = We.get("WEBGL_compressed_texture_pvrtc"),
        null !== t) {
            if (e === n.RGB_PVRTC_4BPPV1_Format)
                return t.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
            if (e === n.RGB_PVRTC_2BPPV1_Format)
                return t.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
            if (e === n.RGBA_PVRTC_4BPPV1_Format)
                return t.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
            if (e === n.RGBA_PVRTC_2BPPV1_Format)
                return t.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG
        }
        if (t = We.get("WEBGL_compressed_texture_etc1"),
        null !== t && e === n.RGB_ETC1_Format)
            return t.COMPRESSED_RGB_ETC1_WEBGL;
        if (t = We.get("EXT_blend_minmax"),
        null !== t) {
            if (e === n.MinEquation)
                return t.MIN_EXT;
            if (e === n.MaxEquation)
                return t.MAX_EXT
        }
        return 0
    }
    console.log("THREE.WebGLRenderer", n.REVISION),
    e = e || {};
    var te = void 0 !== e.canvas ? e.canvas : document.createElement("canvas")
      , ie = void 0 !== e.context ? e.context : null
      , ne = void 0 !== e.alpha && e.alpha
      , re = void 0 === e.depth || e.depth
      , oe = void 0 === e.stencil || e.stencil
      , ae = void 0 !== e.antialias && e.antialias
      , se = void 0 === e.premultipliedAlpha || e.premultipliedAlpha
      , le = void 0 !== e.preserveDrawingBuffer && e.preserveDrawingBuffer
      , he = []
      , ce = []
      , ue = -1
      , de = []
      , pe = -1
      , fe = new Float32Array(8)
      , ge = []
      , me = [];
    this.domElement = te,
    this.context = null,
    this.autoClear = !0,
    this.autoClearColor = !0,
    this.autoClearDepth = !0,
    this.autoClearStencil = !0,
    this.sortObjects = !0,
    this.gammaFactor = 2,
    this.gammaInput = !1,
    this.gammaOutput = !1,
    this.physicallyCorrectLights = !1,
    this.toneMapping = n.LinearToneMapping,
    this.toneMappingExposure = 1,
    this.toneMappingWhitePoint = 1,
    this.maxMorphTargets = 8,
    this.maxMorphNormals = 4,
    this.autoScaleCubemaps = !0;
    var ve = this
      , Ae = null
      , ye = null
      , Ce = null
      , Ie = -1
      , be = ""
      , we = null
      , Ee = new n.Vector4
      , xe = null
      , Te = new n.Vector4
      , Me = 0
      , Se = new n.Color(0)
      , _e = 0
      , Pe = te.width
      , Re = te.height
      , Le = 1
      , Oe = new n.Vector4(0,0,Pe,Re)
      , De = !1
      , Fe = new n.Vector4(0,0,Pe,Re)
      , Ne = new n.Frustum
      , Be = new n.Matrix4
      , ke = new n.Vector3
      , Ue = {
        hash: "",
        ambient: [0, 0, 0],
        directional: [],
        directionalShadowMap: [],
        directionalShadowMatrix: [],
        spot: [],
        spotShadowMap: [],
        spotShadowMatrix: [],
        point: [],
        pointShadowMap: [],
        pointShadowMatrix: [],
        hemi: [],
        shadows: [],
        shadowsPointLight: 0
    }
      , Ve = {
        geometries: 0,
        textures: 0
    }
      , ze = {
        calls: 0,
        vertices: 0,
        faces: 0,
        points: 0
    };
    this.info = {
        render: ze,
        memory: Ve,
        programs: null
    };
    var Ge;
    try {
        var He = {
            alpha: ne,
            depth: re,
            stencil: oe,
            antialias: ae,
            premultipliedAlpha: se,
            preserveDrawingBuffer: le
        };
        if (Ge = ie || te.getContext("webgl", He) || te.getContext("experimental-webgl", He),
        null === Ge)
            throw null !== te.getContext("webgl") ? "Error creating WebGL context with your selected attributes." : "Error creating WebGL context.";
        void 0 === Ge.getShaderPrecisionFormat && (Ge.getShaderPrecisionFormat = function() {
            return {
                rangeMin: 1,
                rangeMax: 1,
                precision: 1
            }
        }
        ),
        te.addEventListener("webglcontextlost", a, !1)
    } catch (e) {
        console.error("THREE.WebGLRenderer: " + e)
    }
    var We = new n.WebGLExtensions(Ge);
    We.get("OES_texture_float"),
    We.get("OES_texture_float_linear"),
    We.get("OES_texture_half_float"),
    We.get("OES_texture_half_float_linear"),
    We.get("OES_standard_derivatives"),
    We.get("ANGLE_instanced_arrays"),
    We.get("OES_element_index_uint") && (n.BufferGeometry.MaxIndex = 4294967296);
    var je = new n.WebGLCapabilities(Ge,We,e)
      , Ye = new n.WebGLState(Ge,We,ee)
      , Xe = new n.WebGLProperties
      , Ze = new n.WebGLObjects(Ge,Xe,this.info)
      , Qe = new n.WebGLPrograms(this,je)
      , qe = new n.WebGLLights;
    this.info.programs = Qe.programs;
    var Ke = new n.WebGLBufferRenderer(Ge,We,ze)
      , Je = new n.WebGLIndexedBufferRenderer(Ge,We,ze);
    r(),
    this.context = Ge,
    this.capabilities = je,
    this.extensions = We,
    this.properties = Xe,
    this.state = Ye;
    var $e = new n.WebGLShadowMap(this,Ue,Ze);
    this.shadowMap = $e;
    var et = new n.SpritePlugin(this,ge)
      , tt = new n.LensFlarePlugin(this,me);
    this.getContext = function() {
        return Ge
    }
    ,
    this.getContextAttributes = function() {
        return Ge.getContextAttributes()
    }
    ,
    this.forceContextLoss = function() {
        We.get("WEBGL_lose_context").loseContext()
    }
    ,
    this.getMaxAnisotropy = function() {
        var e;
        return function() {
            if (void 0 !== e)
                return e;
            var t = We.get("EXT_texture_filter_anisotropic");
            return e = null !== t ? Ge.getParameter(t.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 0
        }
    }(),
    this.getPrecision = function() {
        return je.precision
    }
    ,
    this.getPixelRatio = function() {
        return Le
    }
    ,
    this.setPixelRatio = function(e) {
        void 0 !== e && (Le = e,
        this.setSize(Fe.z, Fe.w, !1))
    }
    ,
    this.getSize = function() {
        return {
            width: Pe,
            height: Re
        }
    }
    ,
    this.setSize = function(e, t, i) {
        Pe = e,
        Re = t,
        te.width = e * Le,
        te.height = t * Le,
        i !== !1 && (te.style.width = e + "px",
        te.style.height = t + "px"),
        this.setViewport(0, 0, e, t)
    }
    ,
    this.setViewport = function(e, t, i, n) {
        Ye.viewport(Fe.set(e, t, i, n))
    }
    ,
    this.setScissor = function(e, t, i, n) {
        Ye.scissor(Oe.set(e, t, i, n))
    }
    ,
    this.setScissorTest = function(e) {
        Ye.setScissorTest(De = e)
    }
    ,
    this.getClearColor = function() {
        return Se
    }
    ,
    this.setClearColor = function(e, t) {
        Se.set(e),
        _e = void 0 !== t ? t : 1,
        i(Se.r, Se.g, Se.b, _e)
    }
    ,
    this.getClearAlpha = function() {
        return _e
    }
    ,
    this.setClearAlpha = function(e) {
        _e = e,
        i(Se.r, Se.g, Se.b, _e)
    }
    ,
    this.clear = function(e, t, i) {
        var n = 0;
        (void 0 === e || e) && (n |= Ge.COLOR_BUFFER_BIT),
        (void 0 === t || t) && (n |= Ge.DEPTH_BUFFER_BIT),
        (void 0 === i || i) && (n |= Ge.STENCIL_BUFFER_BIT),
        Ge.clear(n)
    }
    ,
    this.clearColor = function() {
        this.clear(!0, !1, !1)
    }
    ,
    this.clearDepth = function() {
        this.clear(!1, !0, !1)
    }
    ,
    this.clearStencil = function() {
        this.clear(!1, !1, !0)
    }
    ,
    this.clearTarget = function(e, t, i, n) {
        this.setRenderTarget(e),
        this.clear(t, i, n)
    }
    ,
    this.resetGLState = o,
    this.dispose = function() {
        te.removeEventListener("webglcontextlost", a, !1)
    }
    ,
    this.renderBufferImmediate = function(e, t, i) {
        Ye.initAttributes();
        var r = Xe.get(e);
        e.hasPositions && !r.position && (r.position = Ge.createBuffer()),
        e.hasNormals && !r.normal && (r.normal = Ge.createBuffer()),
        e.hasUvs && !r.uv && (r.uv = Ge.createBuffer()),
        e.hasColors && !r.color && (r.color = Ge.createBuffer());
        var o = t.getAttributes();
        if (e.hasPositions && (Ge.bindBuffer(Ge.ARRAY_BUFFER, r.position),
        Ge.bufferData(Ge.ARRAY_BUFFER, e.positionArray, Ge.DYNAMIC_DRAW),
        Ye.enableAttribute(o.position),
        Ge.vertexAttribPointer(o.position, 3, Ge.FLOAT, !1, 0, 0)),
        e.hasNormals) {
            if (Ge.bindBuffer(Ge.ARRAY_BUFFER, r.normal),
            "MeshPhongMaterial" !== i.type && "MeshStandardMaterial" !== i.type && i.shading === n.FlatShading)
                for (var a = 0, s = 3 * e.count; a < s; a += 9) {
                    var l = e.normalArray
                      , h = (l[a + 0] + l[a + 3] + l[a + 6]) / 3
                      , c = (l[a + 1] + l[a + 4] + l[a + 7]) / 3
                      , u = (l[a + 2] + l[a + 5] + l[a + 8]) / 3;
                    l[a + 0] = h,
                    l[a + 1] = c,
                    l[a + 2] = u,
                    l[a + 3] = h,
                    l[a + 4] = c,
                    l[a + 5] = u,
                    l[a + 6] = h,
                    l[a + 7] = c,
                    l[a + 8] = u
                }
            Ge.bufferData(Ge.ARRAY_BUFFER, e.normalArray, Ge.DYNAMIC_DRAW),
            Ye.enableAttribute(o.normal),
            Ge.vertexAttribPointer(o.normal, 3, Ge.FLOAT, !1, 0, 0)
        }
        e.hasUvs && i.map && (Ge.bindBuffer(Ge.ARRAY_BUFFER, r.uv),
        Ge.bufferData(Ge.ARRAY_BUFFER, e.uvArray, Ge.DYNAMIC_DRAW),
        Ye.enableAttribute(o.uv),
        Ge.vertexAttribPointer(o.uv, 2, Ge.FLOAT, !1, 0, 0)),
        e.hasColors && i.vertexColors !== n.NoColors && (Ge.bindBuffer(Ge.ARRAY_BUFFER, r.color),
        Ge.bufferData(Ge.ARRAY_BUFFER, e.colorArray, Ge.DYNAMIC_DRAW),
        Ye.enableAttribute(o.color),
        Ge.vertexAttribPointer(o.color, 3, Ge.FLOAT, !1, 0, 0)),
        Ye.disableUnusedAttributes(),
        Ge.drawArrays(Ge.TRIANGLES, 0, e.count),
        e.count = 0
    }
    ,
    this.renderBufferDirect = function(e, i, r, o, a, s) {
        b(o);
        var l = E(e, i, o, a)
          , h = !1
          , c = r.id + "_" + l.id + "_" + o.wireframe;
        c !== be && (be = c,
        h = !0);
        var u = a.morphTargetInfluences;
        if (void 0 !== u) {
            for (var d = [], p = 0, m = u.length; p < m; p++) {
                var v = u[p];
                d.push([v, p])
            }
            d.sort(g),
            d.length > 8 && (d.length = 8);
            for (var A = r.morphAttributes, p = 0, m = d.length; p < m; p++) {
                var v = d[p];
                if (fe[p] = v[0],
                0 !== v[0]) {
                    var y = v[1];
                    o.morphTargets === !0 && A.position && r.addAttribute("morphTarget" + p, A.position[y]),
                    o.morphNormals === !0 && A.normal && r.addAttribute("morphNormal" + p, A.normal[y])
                } else
                    o.morphTargets === !0 && r.removeAttribute("morphTarget" + p),
                    o.morphNormals === !0 && r.removeAttribute("morphNormal" + p)
            }
            var C = l.getUniforms();
            null !== C.morphTargetInfluences && Ge.uniform1fv(C.morphTargetInfluences, fe),
            h = !0
        }
        var y = r.index
          , I = r.attributes.position;
        o.wireframe === !0 && (y = Ze.getWireframeAttribute(r));
        var w;
        null !== y ? (w = Je,
        w.setIndex(y)) : w = Ke,
        h && (f(o, l, r),
        null !== y && Ge.bindBuffer(Ge.ELEMENT_ARRAY_BUFFER, Ze.getAttributeBuffer(y)));
        var x = 0
          , T = 1 / 0;
        null !== y ? T = y.count : void 0 !== I && (T = I.count);
        var M = r.drawRange.start
          , S = r.drawRange.count
          , _ = null !== s ? s.start : 0
          , P = null !== s ? s.count : 1 / 0
          , R = Math.max(x, M, _)
          , L = Math.min(x + T, M + S, _ + P) - 1
          , O = Math.max(0, L - R + 1);
        if (a instanceof n.Mesh)
            if (o.wireframe === !0)
                Ye.setLineWidth(o.wireframeLinewidth * t()),
                w.setMode(Ge.LINES);
            else
                switch (a.drawMode) {
                case n.TrianglesDrawMode:
                    w.setMode(Ge.TRIANGLES);
                    break;
                case n.TriangleStripDrawMode:
                    w.setMode(Ge.TRIANGLE_STRIP);
                    break;
                case n.TriangleFanDrawMode:
                    w.setMode(Ge.TRIANGLE_FAN)
                }
        else if (a instanceof n.Line) {
            var D = o.linewidth;
            void 0 === D && (D = 1),
            Ye.setLineWidth(D * t()),
            a instanceof n.LineSegments ? w.setMode(Ge.LINES) : w.setMode(Ge.LINE_STRIP)
        } else
            a instanceof n.Points && w.setMode(Ge.POINTS);
        r instanceof n.InstancedBufferGeometry ? r.maxInstancedCount > 0 && w.renderInstances(r, R, O) : w.render(R, O)
    }
    ,
    this.render = function(e, t, i, r) {
        if (t instanceof n.Camera == !1)
            return void console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");
        var o = e.fog;
        if (be = "",
        Ie = -1,
        we = null,
        e.autoUpdate === !0 && e.updateMatrixWorld(),
        null === t.parent && t.updateMatrixWorld(),
        t.matrixWorldInverse.getInverse(t.matrixWorld),
        Be.multiplyMatrices(t.projectionMatrix, t.matrixWorldInverse),
        Ne.setFromMatrix(Be),
        he.length = 0,
        ue = -1,
        pe = -1,
        ge.length = 0,
        me.length = 0,
        y(e, t),
        ce.length = ue + 1,
        de.length = pe + 1,
        ve.sortObjects === !0 && (ce.sort(m),
        de.sort(v)),
        U(he, t),
        $e.render(e, t),
        ze.calls = 0,
        ze.vertices = 0,
        ze.faces = 0,
        ze.points = 0,
        void 0 === i && (i = null),
        this.setRenderTarget(i),
        (this.autoClear || r) && this.clear(this.autoClearColor, this.autoClearDepth, this.autoClearStencil),
        e.overrideMaterial) {
            var a = e.overrideMaterial;
            C(ce, t, o, a),
            C(de, t, o, a)
        } else
            Ye.setBlending(n.NoBlending),
            C(ce, t, o),
            C(de, t, o);
        if (et.render(e, t),
        tt.render(e, t, Te),
        i) {
            var s = i.texture;
            s.generateMipmaps && H(i) && s.minFilter !== n.NearestFilter && s.minFilter !== n.LinearFilter && J(i)
        }
        Ye.setDepthTest(!0),
        Ye.setDepthWrite(!0),
        Ye.setColorWrite(!0)
    }
    ,
    this.setFaceCulling = function(e, t) {
        e === n.CullFaceNone ? Ye.disable(Ge.CULL_FACE) : (t === n.FrontFaceDirectionCW ? Ge.frontFace(Ge.CW) : Ge.frontFace(Ge.CCW),
        e === n.CullFaceBack ? Ge.cullFace(Ge.BACK) : e === n.CullFaceFront ? Ge.cullFace(Ge.FRONT) : Ge.cullFace(Ge.FRONT_AND_BACK),
        Ye.enable(Ge.CULL_FACE))
    }
    ,
    this.setTexture = function(e, t) {
        var i = Xe.get(e);
        if (e.version > 0 && i.__version !== e.version) {
            var n = e.image;
            return void 0 === n ? void console.warn("THREE.WebGLRenderer: Texture marked for update but image is undefined", e) : n.complete === !1 ? void console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete", e) : void z(i, e, t)
        }
        Ye.activeTexture(Ge.TEXTURE0 + t),
        Ye.bindTexture(Ge.TEXTURE_2D, i.__webglTexture)
    }
    ,
    this.getCurrentRenderTarget = function() {
        return ye
    }
    ,
    this.setRenderTarget = function(e) {
        ye = e,
        e && void 0 === Xe.get(e).__webglFramebuffer && K(e);
        var t, i = e instanceof n.WebGLRenderTargetCube;
        if (e) {
            var r = Xe.get(e);
            t = i ? r.__webglFramebuffer[e.activeCubeFace] : r.__webglFramebuffer,
            Ee.copy(e.scissor),
            xe = e.scissorTest,
            Te.copy(e.viewport)
        } else
            t = null,
            Ee.copy(Oe).multiplyScalar(Le),
            xe = De,
            Te.copy(Fe).multiplyScalar(Le);
        if (Ce !== t && (Ge.bindFramebuffer(Ge.FRAMEBUFFER, t),
        Ce = t),
        Ye.scissor(Ee),
        Ye.setScissorTest(xe),
        Ye.viewport(Te),
        i) {
            var o = Xe.get(e.texture);
            Ge.framebufferTexture2D(Ge.FRAMEBUFFER, Ge.COLOR_ATTACHMENT0, Ge.TEXTURE_CUBE_MAP_POSITIVE_X + e.activeCubeFace, o.__webglTexture, e.activeMipMapLevel)
        }
    }
    ,
    this.readRenderTargetPixels = function(e, t, i, r, o, a) {
        if (e instanceof n.WebGLRenderTarget == !1)
            return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
        var s = Xe.get(e).__webglFramebuffer;
        if (s) {
            var l = !1;
            s !== Ce && (Ge.bindFramebuffer(Ge.FRAMEBUFFER, s),
            l = !0);
            try {
                var h = e.texture;
                if (h.format !== n.RGBAFormat && ee(h.format) !== Ge.getParameter(Ge.IMPLEMENTATION_COLOR_READ_FORMAT))
                    return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");
                if (!(h.type === n.UnsignedByteType || ee(h.type) === Ge.getParameter(Ge.IMPLEMENTATION_COLOR_READ_TYPE) || h.type === n.FloatType && We.get("WEBGL_color_buffer_float") || h.type === n.HalfFloatType && We.get("EXT_color_buffer_half_float")))
                    return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");
                Ge.checkFramebufferStatus(Ge.FRAMEBUFFER) === Ge.FRAMEBUFFER_COMPLETE ? Ge.readPixels(t, i, r, o, ee(h.format), ee(h.type), a) : console.error("THREE.WebGLRenderer.readRenderTargetPixels: readPixels from renderTarget failed. Framebuffer not complete.")
            } finally {
                l && Ge.bindFramebuffer(Ge.FRAMEBUFFER, Ce)
            }
        }
    }
}
,
n.WebGLRenderTarget = function(e, t, i) {
    this.uuid = n.Math.generateUUID(),
    this.width = e,
    this.height = t,
    this.scissor = new n.Vector4(0,0,e,t),
    this.scissorTest = !1,
    this.viewport = new n.Vector4(0,0,e,t),
    i = i || {},
    void 0 === i.minFilter && (i.minFilter = n.LinearFilter),
    this.texture = new n.Texture(void 0,void 0,i.wrapS,i.wrapT,i.magFilter,i.minFilter,i.format,i.type,i.anisotropy),
    this.depthBuffer = void 0 === i.depthBuffer || i.depthBuffer,
    this.stencilBuffer = void 0 === i.stencilBuffer || i.stencilBuffer
}
,
n.WebGLRenderTarget.prototype = {
    constructor: n.WebGLRenderTarget,
    setSize: function(e, t) {
        this.width === e && this.height === t || (this.width = e,
        this.height = t,
        this.dispose()),
        this.viewport.set(0, 0, e, t),
        this.scissor.set(0, 0, e, t)
    },
    clone: function() {
        return (new this.constructor).copy(this)
    },
    copy: function(e) {
        return this.width = e.width,
        this.height = e.height,
        this.viewport.copy(e.viewport),
        this.texture = e.texture.clone(),
        this.depthBuffer = e.depthBuffer,
        this.stencilBuffer = e.stencilBuffer,
        this
    },
    dispose: function() {
        this.dispatchEvent({
            type: "dispose"
        })
    }
},
n.EventDispatcher.prototype.apply(n.WebGLRenderTarget.prototype),
n.WebGLRenderTargetCube = function(e, t, i) {
    n.WebGLRenderTarget.call(this, e, t, i),
    this.activeCubeFace = 0,
    this.activeMipMapLevel = 0
}
,
n.WebGLRenderTargetCube.prototype = Object.create(n.WebGLRenderTarget.prototype),
n.WebGLRenderTargetCube.prototype.constructor = n.WebGLRenderTargetCube,
n.WebGLBufferRenderer = function(e, t, i) {
    function r(e) {
        s = e
    }
    function o(t, n) {
        e.drawArrays(s, t, n),
        i.calls++,
        i.vertices += n,
        s === e.TRIANGLES && (i.faces += n / 3)
    }
    function a(r) {
        var o = t.get("ANGLE_instanced_arrays");
        if (null === o)
            return void console.error("THREE.WebGLBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");
        var a = r.attributes.position
          , l = 0;
        a instanceof n.InterleavedBufferAttribute ? (l = a.data.count,
        o.drawArraysInstancedANGLE(s, 0, l, r.maxInstancedCount)) : (l = a.count,
        o.drawArraysInstancedANGLE(s, 0, l, r.maxInstancedCount)),
        i.calls++,
        i.vertices += l * r.maxInstancedCount,
        s === e.TRIANGLES && (i.faces += r.maxInstancedCount * l / 3)
    }
    var s;
    this.setMode = r,
    this.render = o,
    this.renderInstances = a
}
,
n.WebGLIndexedBufferRenderer = function(e, t, i) {
    function n(e) {
        s = e
    }
    function r(i) {
        i.array instanceof Uint32Array && t.get("OES_element_index_uint") ? (l = e.UNSIGNED_INT,
        h = 4) : (l = e.UNSIGNED_SHORT,
        h = 2)
    }
    function o(t, n) {
        e.drawElements(s, n, l, t * h),
        i.calls++,
        i.vertices += n,
        s === e.TRIANGLES && (i.faces += n / 3)
    }
    function a(n, r, o) {
        var a = t.get("ANGLE_instanced_arrays");
        return null === a ? void console.error("THREE.WebGLBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.") : (a.drawElementsInstancedANGLE(s, o, l, r * h, n.maxInstancedCount),
        i.calls++,
        i.vertices += o * n.maxInstancedCount,
        void (s === e.TRIANGLES && (i.faces += n.maxInstancedCount * o / 3)))
    }
    var s, l, h;
    this.setMode = n,
    this.setIndex = r,
    this.render = o,
    this.renderInstances = a
}
,
n.WebGLExtensions = function(e) {
    var t = {};
    this.get = function(i) {
        if (void 0 !== t[i])
            return t[i];
        var n;
        switch (i) {
        case "EXT_texture_filter_anisotropic":
            n = e.getExtension("EXT_texture_filter_anisotropic") || e.getExtension("MOZ_EXT_texture_filter_anisotropic") || e.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
            break;
        case "WEBGL_compressed_texture_s3tc":
            n = e.getExtension("WEBGL_compressed_texture_s3tc") || e.getExtension("MOZ_WEBGL_compressed_texture_s3tc") || e.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");
            break;
        case "WEBGL_compressed_texture_pvrtc":
            n = e.getExtension("WEBGL_compressed_texture_pvrtc") || e.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");
            break;
        case "WEBGL_compressed_texture_etc1":
            n = e.getExtension("WEBGL_compressed_texture_etc1");
            break;
        default:
            n = e.getExtension(i)
        }
        return null === n && console.warn("THREE.WebGLRenderer: " + i + " extension not supported."),
        t[i] = n,
        n
    }
}
,
n.WebGLCapabilities = function(e, t, i) {
    function n(t) {
        if ("highp" === t) {
            if (e.getShaderPrecisionFormat(e.VERTEX_SHADER, e.HIGH_FLOAT).precision > 0 && e.getShaderPrecisionFormat(e.FRAGMENT_SHADER, e.HIGH_FLOAT).precision > 0)
                return "highp";
            t = "mediump"
        }
        return "mediump" === t && e.getShaderPrecisionFormat(e.VERTEX_SHADER, e.MEDIUM_FLOAT).precision > 0 && e.getShaderPrecisionFormat(e.FRAGMENT_SHADER, e.MEDIUM_FLOAT).precision > 0 ? "mediump" : "lowp"
    }
    this.getMaxPrecision = n,
    this.precision = void 0 !== i.precision ? i.precision : "highp",
    this.logarithmicDepthBuffer = void 0 !== i.logarithmicDepthBuffer && i.logarithmicDepthBuffer,
    this.maxTextures = e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS),
    this.maxVertexTextures = e.getParameter(e.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
    this.maxTextureSize = e.getParameter(e.MAX_TEXTURE_SIZE),
    this.maxCubemapSize = e.getParameter(e.MAX_CUBE_MAP_TEXTURE_SIZE),
    this.maxAttributes = e.getParameter(e.MAX_VERTEX_ATTRIBS),
    this.maxVertexUniforms = e.getParameter(e.MAX_VERTEX_UNIFORM_VECTORS),
    this.maxVaryings = e.getParameter(e.MAX_VARYING_VECTORS),
    this.maxFragmentUniforms = e.getParameter(e.MAX_FRAGMENT_UNIFORM_VECTORS),
    this.vertexTextures = this.maxVertexTextures > 0,
    this.floatFragmentTextures = !!t.get("OES_texture_float"),
    this.floatVertexTextures = this.vertexTextures && this.floatFragmentTextures;
    var r = n(this.precision);
    r !== this.precision && (console.warn("THREE.WebGLRenderer:", this.precision, "not supported, using", r, "instead."),
    this.precision = r),
    this.logarithmicDepthBuffer && (this.logarithmicDepthBuffer = !!t.get("EXT_frag_depth"))
}
,
n.WebGLGeometries = function(e, t, i) {
    function r(e) {
        var t = e.geometry;
        if (void 0 !== c[t.id])
            return c[t.id];
        t.addEventListener("dispose", o);
        var r;
        return t instanceof n.BufferGeometry ? r = t : t instanceof n.Geometry && (void 0 === t._bufferGeometry && (t._bufferGeometry = (new n.BufferGeometry).setFromObject(e)),
        r = t._bufferGeometry),
        c[t.id] = r,
        i.memory.geometries++,
        r
    }
    function o(e) {
        var n = e.target
          , r = c[n.id];
        null !== r.index && s(r.index),
        l(r.attributes),
        n.removeEventListener("dispose", o),
        delete c[n.id];
        var a = t.get(n);
        a.wireframe && s(a.wireframe),
        t.delete(n);
        var h = t.get(r);
        h.wireframe && s(h.wireframe),
        t.delete(r),
        i.memory.geometries--
    }
    function a(e) {
        return e instanceof n.InterleavedBufferAttribute ? t.get(e.data).__webglBuffer : t.get(e).__webglBuffer
    }
    function s(t) {
        var i = a(t);
        void 0 !== i && (e.deleteBuffer(i),
        h(t))
    }
    function l(e) {
        for (var t in e)
            s(e[t])
    }
    function h(e) {
        e instanceof n.InterleavedBufferAttribute ? t.delete(e.data) : t.delete(e)
    }
    var c = {};
    this.get = r
}
,
n.WebGLLights = function() {
    var e = {};
    this.get = function(t) {
        if (void 0 !== e[t.id])
            return e[t.id];
        var i;
        switch (t.type) {
        case "DirectionalLight":
            i = {
                direction: new n.Vector3,
                color: new n.Color,
                shadow: !1,
                shadowBias: 0,
                shadowRadius: 1,
                shadowMapSize: new n.Vector2
            };
            break;
        case "SpotLight":
            i = {
                position: new n.Vector3,
                direction: new n.Vector3,
                color: new n.Color,
                distance: 0,
                coneCos: 0,
                penumbraCos: 0,
                decay: 0,
                shadow: !1,
                shadowBias: 0,
                shadowRadius: 1,
                shadowMapSize: new n.Vector2
            };
            break;
        case "PointLight":
            i = {
                position: new n.Vector3,
                color: new n.Color,
                distance: 0,
                decay: 0,
                shadow: !1,
                shadowBias: 0,
                shadowRadius: 1,
                shadowMapSize: new n.Vector2
            };
            break;
        case "HemisphereLight":
            i = {
                direction: new n.Vector3,
                skyColor: new n.Color,
                groundColor: new n.Color
            }
        }
        return e[t.id] = i,
        i
    }
}
,
n.WebGLObjects = function(e, t, i) {
    function r(t) {
        var i = u.get(t);
        t.geometry instanceof n.Geometry && i.updateFromObject(t);
        var r = i.index
          , a = i.attributes;
        null !== r && o(r, e.ELEMENT_ARRAY_BUFFER);
        for (var s in a)
            o(a[s], e.ARRAY_BUFFER);
        var l = i.morphAttributes;
        for (var s in l)
            for (var h = l[s], c = 0, d = h.length; c < d; c++)
                o(h[c], e.ARRAY_BUFFER);
        return i
    }
    function o(e, i) {
        var r = e instanceof n.InterleavedBufferAttribute ? e.data : e
          , o = t.get(r);
        void 0 === o.__webglBuffer ? a(o, r, i) : o.version !== r.version && s(o, r, i)
    }
    function a(t, i, n) {
        t.__webglBuffer = e.createBuffer(),
        e.bindBuffer(n, t.__webglBuffer);
        var r = i.dynamic ? e.DYNAMIC_DRAW : e.STATIC_DRAW;
        e.bufferData(n, i.array, r),
        t.version = i.version
    }
    function s(t, i, n) {
        e.bindBuffer(n, t.__webglBuffer),
        i.dynamic === !1 || i.updateRange.count === -1 ? e.bufferSubData(n, 0, i.array) : 0 === i.updateRange.count ? console.error("THREE.WebGLObjects.updateBuffer: dynamic THREE.BufferAttribute marked as needsUpdate but updateRange.count is 0, ensure you are using set methods or updating manually.") : (e.bufferSubData(n, i.updateRange.offset * i.array.BYTES_PER_ELEMENT, i.array.subarray(i.updateRange.offset, i.updateRange.offset + i.updateRange.count)),
        i.updateRange.count = 0),
        t.version = i.version
    }
    function l(e) {
        return e instanceof n.InterleavedBufferAttribute ? t.get(e.data).__webglBuffer : t.get(e).__webglBuffer
    }
    function h(i) {
        var r = t.get(i);
        if (void 0 !== r.wireframe)
            return r.wireframe;
        var a = []
          , s = i.index
          , l = i.attributes
          , h = l.position;
        if (null !== s)
            for (var u = {}, d = s.array, p = 0, f = d.length; p < f; p += 3) {
                var g = d[p + 0]
                  , m = d[p + 1]
                  , v = d[p + 2];
                c(u, g, m) && a.push(g, m),
                c(u, m, v) && a.push(m, v),
                c(u, v, g) && a.push(v, g)
            }
        else
            for (var d = l.position.array, p = 0, f = d.length / 3 - 1; p < f; p += 3) {
                var g = p + 0
                  , m = p + 1
                  , v = p + 2;
                a.push(g, m, m, v, v, g)
            }
        var A = h.count > 65535 ? Uint32Array : Uint16Array
          , y = new n.BufferAttribute(new A(a),1);
        return o(y, e.ELEMENT_ARRAY_BUFFER),
        r.wireframe = y,
        y
    }
    function c(e, t, i) {
        if (t > i) {
            var n = t;
            t = i,
            i = n
        }
        var r = e[t];
        return void 0 === r ? (e[t] = [i],
        !0) : r.indexOf(i) === -1 && (r.push(i),
        !0)
    }
    var u = new n.WebGLGeometries(e,t,i);
    this.getAttributeBuffer = l,
    this.getWireframeAttribute = h,
    this.update = r
}
,
n.WebGLProgram = function() {
    function e(e) {
        switch (e) {
        case n.LinearEncoding:
            return ["Linear", "( value )"];
        case n.sRGBEncoding:
            return ["sRGB", "( value )"];
        case n.RGBEEncoding:
            return ["RGBE", "( value )"];
        case n.RGBM7Encoding:
            return ["RGBM", "( value, 7.0 )"];
        case n.RGBM16Encoding:
            return ["RGBM", "( value, 16.0 )"];
        case n.RGBDEncoding:
            return ["RGBD", "( value, 256.0 )"];
        case n.GammaEncoding:
            return ["Gamma", "( value, float( GAMMA_FACTOR ) )"];
        default:
            throw new Error("unsupported encoding: " + e)
        }
    }
    function t(t, i) {
        var n = e(i);
        return "vec4 " + t + "( vec4 value ) { return " + n[0] + "ToLinear" + n[1] + "; }"
    }
    function i(t, i) {
        var n = e(i);
        return "vec4 " + t + "( vec4 value ) { return LinearTo" + n[0] + n[1] + "; }"
    }
    function r(e, t) {
        var i;
        switch (t) {
        case n.LinearToneMapping:
            i = "Linear";
            break;
        case n.ReinhardToneMapping:
            i = "Reinhard";
            break;
        case n.Uncharted2ToneMapping:
            i = "Uncharted2";
            break;
        case n.CineonToneMapping:
            i = "OptimizedCineon";
            break;
        default:
            throw new Error("unsupported toneMapping: " + t)
        }
        return "vec3 " + e + "( vec3 color ) { return " + i + "ToneMapping( color ); }"
    }
    function o(e, t, i) {
        e = e || {};
        var n = [e.derivatives || t.envMapCubeUV || t.bumpMap || t.normalMap || t.flatShading ? "#extension GL_OES_standard_derivatives : enable" : "", (e.fragDepth || t.logarithmicDepthBuffer) && i.get("EXT_frag_depth") ? "#extension GL_EXT_frag_depth : enable" : "", e.drawBuffers && i.get("WEBGL_draw_buffers") ? "#extension GL_EXT_draw_buffers : require" : "", (e.shaderTextureLOD || t.envMap) && i.get("EXT_shader_texture_lod") ? "#extension GL_EXT_shader_texture_lod : enable" : ""];
        return n.filter(h).join("\n")
    }
    function a(e) {
        var t = [];
        for (var i in e) {
            var n = e[i];
            n !== !1 && t.push("#define " + i + " " + n)
        }
        return t.join("\n")
    }
    function s(e, t, i) {
        for (var n = {}, r = e.getProgramParameter(t, e.ACTIVE_UNIFORMS), o = 0; o < r; o++) {
            var a = e.getActiveUniform(t, o)
              , s = a.name
              , l = e.getUniformLocation(t, s)
              , h = f.exec(s);
            if (h) {
                var c = h[1]
                  , u = h[2]
                  , d = n[c];
                d || (d = n[c] = {}),
                d[u] = l
            } else if (h = g.exec(s)) {
                var p = h[1]
                  , v = h[2]
                  , A = h[3]
                  , y = n[p];
                y || (y = n[p] = []);
                var C = y[v];
                C || (C = y[v] = {}),
                C[A] = l
            } else if (h = m.exec(s)) {
                var p = h[1];
                n[p] = l
            } else
                n[s] = l
        }
        return n
    }
    function l(e, t, i) {
        for (var n = {}, r = e.getProgramParameter(t, e.ACTIVE_ATTRIBUTES), o = 0; o < r; o++) {
            var a = e.getActiveAttrib(t, o)
              , s = a.name;
            n[s] = e.getAttribLocation(t, s)
        }
        return n
    }
    function h(e) {
        return "" !== e
    }
    function c(e, t) {
        return e.replace(/NUM_DIR_LIGHTS/g, t.numDirLights).replace(/NUM_SPOT_LIGHTS/g, t.numSpotLights).replace(/NUM_POINT_LIGHTS/g, t.numPointLights).replace(/NUM_HEMI_LIGHTS/g, t.numHemiLights)
    }
    function u(e) {
        function t(e, t) {
            var i = n.ShaderChunk[t];
            if (void 0 === i)
                throw new Error("Can not resolve #include <" + t + ">");
            return u(i)
        }
        var i = /#include +<([\w\d.]+)>/g;
        return e.replace(i, t)
    }
    function d(e) {
        function t(e, t, i, n) {
            for (var r = "", o = parseInt(t); o < parseInt(i); o++)
                r += n.replace(/\[ i \]/g, "[ " + o + " ]");
            return r
        }
        var i = /for \( int i \= (\d+)\; i < (\d+)\; i \+\+ \) \{([\s\S]+?)(?=\})\}/g;
        return e.replace(i, t)
    }
    var p = 0
      , f = /^([\w\d_]+)\.([\w\d_]+)$/
      , g = /^([\w\d_]+)\[(\d+)\]\.([\w\d_]+)$/
      , m = /^([\w\d_]+)\[0\]$/;
    return function(e, f, g, m) {
        var v = e.context
          , A = g.extensions
          , y = g.defines
          , C = g.__webglShader.vertexShader
          , I = g.__webglShader.fragmentShader
          , b = "SHADOWMAP_TYPE_BASIC";
        m.shadowMapType === n.PCFShadowMap ? b = "SHADOWMAP_TYPE_PCF" : m.shadowMapType === n.PCFSoftShadowMap && (b = "SHADOWMAP_TYPE_PCF_SOFT");
        var w = "ENVMAP_TYPE_CUBE"
          , E = "ENVMAP_MODE_REFLECTION"
          , x = "ENVMAP_BLENDING_MULTIPLY";
        if (m.envMap) {
            switch (g.envMap.mapping) {
            case n.CubeReflectionMapping:
            case n.CubeRefractionMapping:
                w = "ENVMAP_TYPE_CUBE";
                break;
            case n.CubeUVReflectionMapping:
            case n.CubeUVRefractionMapping:
                w = "ENVMAP_TYPE_CUBE_UV";
                break;
            case n.EquirectangularReflectionMapping:
            case n.EquirectangularRefractionMapping:
                w = "ENVMAP_TYPE_EQUIREC";
                break;
            case n.SphericalReflectionMapping:
                w = "ENVMAP_TYPE_SPHERE"
            }
            switch (g.envMap.mapping) {
            case n.CubeRefractionMapping:
            case n.EquirectangularRefractionMapping:
                E = "ENVMAP_MODE_REFRACTION"
            }
            switch (g.combine) {
            case n.MultiplyOperation:
                x = "ENVMAP_BLENDING_MULTIPLY";
                break;
            case n.MixOperation:
                x = "ENVMAP_BLENDING_MIX";
                break;
            case n.AddOperation:
                x = "ENVMAP_BLENDING_ADD"
            }
        }
        var T, M, S = e.gammaFactor > 0 ? e.gammaFactor : 1, _ = o(A, m, e.extensions), P = a(y), R = v.createProgram();
        g instanceof n.RawShaderMaterial ? (T = "",
        M = "") : (T = ["precision " + m.precision + " float;", "precision " + m.precision + " int;", "#define SHADER_NAME " + g.__webglShader.name, P, m.supportsVertexTextures ? "#define VERTEX_TEXTURES" : "", "#define GAMMA_FACTOR " + S, "#define MAX_BONES " + m.maxBones, m.map ? "#define USE_MAP" : "", m.envMap ? "#define USE_ENVMAP" : "", m.envMap ? "#define " + E : "", m.lightMap ? "#define USE_LIGHTMAP" : "", m.aoMap ? "#define USE_AOMAP" : "", m.emissiveMap ? "#define USE_EMISSIVEMAP" : "", m.bumpMap ? "#define USE_BUMPMAP" : "", m.normalMap ? "#define USE_NORMALMAP" : "", m.displacementMap && m.supportsVertexTextures ? "#define USE_DISPLACEMENTMAP" : "", m.specularMap ? "#define USE_SPECULARMAP" : "", m.roughnessMap ? "#define USE_ROUGHNESSMAP" : "", m.metalnessMap ? "#define USE_METALNESSMAP" : "", m.alphaMap ? "#define USE_ALPHAMAP" : "", m.vertexColors ? "#define USE_COLOR" : "", m.flatShading ? "#define FLAT_SHADED" : "", m.skinning ? "#define USE_SKINNING" : "", m.useVertexTexture ? "#define BONE_TEXTURE" : "", m.morphTargets ? "#define USE_MORPHTARGETS" : "", m.morphNormals && m.flatShading === !1 ? "#define USE_MORPHNORMALS" : "", m.doubleSided ? "#define DOUBLE_SIDED" : "", m.flipSided ? "#define FLIP_SIDED" : "", m.shadowMapEnabled ? "#define USE_SHADOWMAP" : "", m.shadowMapEnabled ? "#define " + b : "", m.pointLightShadows > 0 ? "#define POINT_LIGHT_SHADOWS" : "", m.sizeAttenuation ? "#define USE_SIZEATTENUATION" : "", m.logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "", m.logarithmicDepthBuffer && e.extensions.get("EXT_frag_depth") ? "#define USE_LOGDEPTHBUF_EXT" : "", "uniform mat4 modelMatrix;", "uniform mat4 modelViewMatrix;", "uniform mat4 projectionMatrix;", "uniform mat4 viewMatrix;", "uniform mat3 normalMatrix;", "uniform vec3 cameraPosition;", "attribute vec3 position;", "attribute vec3 normal;", "attribute vec2 uv;", "#ifdef USE_COLOR", "\tattribute vec3 color;", "#endif", "#ifdef USE_MORPHTARGETS", "\tattribute vec3 morphTarget0;", "\tattribute vec3 morphTarget1;", "\tattribute vec3 morphTarget2;", "\tattribute vec3 morphTarget3;", "\t#ifdef USE_MORPHNORMALS", "\t\tattribute vec3 morphNormal0;", "\t\tattribute vec3 morphNormal1;", "\t\tattribute vec3 morphNormal2;", "\t\tattribute vec3 morphNormal3;", "\t#else", "\t\tattribute vec3 morphTarget4;", "\t\tattribute vec3 morphTarget5;", "\t\tattribute vec3 morphTarget6;", "\t\tattribute vec3 morphTarget7;", "\t#endif", "#endif", "#ifdef USE_SKINNING", "\tattribute vec4 skinIndex;", "\tattribute vec4 skinWeight;", "#endif", "\n"].filter(h).join("\n"),
        M = [_, "precision " + m.precision + " float;", "precision " + m.precision + " int;", "#define SHADER_NAME " + g.__webglShader.name, P, m.alphaTest ? "#define ALPHATEST " + m.alphaTest : "", "#define GAMMA_FACTOR " + S, m.useFog && m.fog ? "#define USE_FOG" : "", m.useFog && m.fogExp ? "#define FOG_EXP2" : "", m.map ? "#define USE_MAP" : "", m.envMap ? "#define USE_ENVMAP" : "", m.envMap ? "#define " + w : "", m.envMap ? "#define " + E : "", m.envMap ? "#define " + x : "", m.lightMap ? "#define USE_LIGHTMAP" : "", m.aoMap ? "#define USE_AOMAP" : "", m.emissiveMap ? "#define USE_EMISSIVEMAP" : "", m.bumpMap ? "#define USE_BUMPMAP" : "", m.normalMap ? "#define USE_NORMALMAP" : "", m.specularMap ? "#define USE_SPECULARMAP" : "", m.roughnessMap ? "#define USE_ROUGHNESSMAP" : "", m.metalnessMap ? "#define USE_METALNESSMAP" : "", m.alphaMap ? "#define USE_ALPHAMAP" : "", m.vertexColors ? "#define USE_COLOR" : "", m.flatShading ? "#define FLAT_SHADED" : "", m.doubleSided ? "#define DOUBLE_SIDED" : "", m.flipSided ? "#define FLIP_SIDED" : "", m.shadowMapEnabled ? "#define USE_SHADOWMAP" : "", m.shadowMapEnabled ? "#define " + b : "", m.pointLightShadows > 0 ? "#define POINT_LIGHT_SHADOWS" : "", m.premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "", m.physicallyCorrectLights ? "#define PHYSICALLY_CORRECT_LIGHTS" : "", m.logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "", m.logarithmicDepthBuffer && e.extensions.get("EXT_frag_depth") ? "#define USE_LOGDEPTHBUF_EXT" : "", m.envMap && e.extensions.get("EXT_shader_texture_lod") ? "#define TEXTURE_LOD_EXT" : "", "uniform mat4 viewMatrix;", "uniform vec3 cameraPosition;", m.toneMapping !== n.NoToneMapping ? "#define TONE_MAPPING" : "", m.toneMapping !== n.NoToneMapping ? n.ShaderChunk.tonemapping_pars_fragment : "", m.toneMapping !== n.NoToneMapping ? r("toneMapping", m.toneMapping) : "", m.outputEncoding || m.mapEncoding || m.envMapEncoding || m.emissiveMapEncoding ? n.ShaderChunk.encodings_pars_fragment : "", m.mapEncoding ? t("mapTexelToLinear", m.mapEncoding) : "", m.envMapEncoding ? t("envMapTexelToLinear", m.envMapEncoding) : "", m.emissiveMapEncoding ? t("emissiveMapTexelToLinear", m.emissiveMapEncoding) : "", m.outputEncoding ? i("linearToOutputTexel", m.outputEncoding) : "", "\n"].filter(h).join("\n")),
        C = u(C, m),
        C = c(C, m),
        I = u(I, m),
        I = c(I, m),
        g instanceof n.ShaderMaterial == !1 && (C = d(C),
        I = d(I));
        var L = T + C
          , O = M + I
          , D = n.WebGLShader(v, v.VERTEX_SHADER, L)
          , F = n.WebGLShader(v, v.FRAGMENT_SHADER, O);
        v.attachShader(R, D),
        v.attachShader(R, F),
        void 0 !== g.index0AttributeName ? v.bindAttribLocation(R, 0, g.index0AttributeName) : m.morphTargets === !0 && v.bindAttribLocation(R, 0, "position"),
        v.linkProgram(R);
        var N = v.getProgramInfoLog(R)
          , B = v.getShaderInfoLog(D)
          , k = v.getShaderInfoLog(F)
          , U = !0
          , V = !0;
        v.getProgramParameter(R, v.LINK_STATUS) === !1 ? (U = !1,
        console.error("THREE.WebGLProgram: shader error: ", v.getError(), "gl.VALIDATE_STATUS", v.getProgramParameter(R, v.VALIDATE_STATUS), "gl.getProgramInfoLog", N, B, k)) : "" !== N ? console.warn("THREE.WebGLProgram: gl.getProgramInfoLog()", N) : "" !== B && "" !== k || (V = !1),
        V && (this.diagnostics = {
            runnable: U,
            material: g,
            programLog: N,
            vertexShader: {
                log: B,
                prefix: T
            },
            fragmentShader: {
                log: k,
                prefix: M
            }
        }),
        v.deleteShader(D),
        v.deleteShader(F);
        var z;
        this.getUniforms = function() {
            return void 0 === z && (z = s(v, R)),
            z
        }
        ;
        var G;
        return this.getAttributes = function() {
            return void 0 === G && (G = l(v, R)),
            G
        }
        ,
        this.destroy = function() {
            v.deleteProgram(R),
            this.program = void 0
        }
        ,
        Object.defineProperties(this, {
            uniforms: {
                get: function() {
                    return console.warn("THREE.WebGLProgram: .uniforms is now .getUniforms()."),
                    this.getUniforms()
                }
            },
            attributes: {
                get: function() {
                    return console.warn("THREE.WebGLProgram: .attributes is now .getAttributes()."),
                    this.getAttributes()
                }
            }
        }),
        this.id = p++,
        this.code = f,
        this.usedTimes = 1,
        this.program = R,
        this.vertexShader = D,
        this.fragmentShader = F,
        this
    }
}(),
n.WebGLPrograms = function(e, t) {
    function i(e) {
        if (t.floatVertexTextures && e && e.skeleton && e.skeleton.useVertexTexture)
            return 1024;
        var i = t.maxVertexUniforms
          , r = Math.floor((i - 20) / 4)
          , o = r;
        return void 0 !== e && e instanceof n.SkinnedMesh && (o = Math.min(e.skeleton.bones.length, o),
        o < e.skeleton.bones.length && console.warn("WebGLRenderer: too many bones - " + e.skeleton.bones.length + ", this GPU supports just " + o + " (try OpenGL instead of ANGLE)")),
        o
    }
    function r(e, t) {
        var i;
        return e ? e instanceof n.Texture ? i = e.encoding : e instanceof n.WebGLRenderTarget && (i = e.texture.encoding) : i = n.LinearEncoding,
        i === n.LinearEncoding && t && (i = n.GammaEncoding),
        i
    }
    var o = []
      , a = {
        MeshDepthMaterial: "depth",
        MeshNormalMaterial: "normal",
        MeshBasicMaterial: "basic",
        MeshLambertMaterial: "lambert",
        MeshPhongMaterial: "phong",
        MeshStandardMaterial: "standard",
        LineBasicMaterial: "basic",
        LineDashedMaterial: "dashed",
        PointsMaterial: "points"
    }
      , s = ["precision", "supportsVertexTextures", "map", "mapEncoding", "envMap", "envMapMode", "envMapEncoding", "lightMap", "aoMap", "emissiveMap", "emissiveMapEncoding", "bumpMap", "normalMap", "displacementMap", "specularMap", "roughnessMap", "metalnessMap", "alphaMap", "combine", "vertexColors", "fog", "useFog", "fogExp", "flatShading", "sizeAttenuation", "logarithmicDepthBuffer", "skinning", "maxBones", "useVertexTexture", "morphTargets", "morphNormals", "maxMorphTargets", "maxMorphNormals", "premultipliedAlpha", "numDirLights", "numPointLights", "numSpotLights", "numHemiLights", "shadowMapEnabled", "pointLightShadows", "toneMapping", "physicallyCorrectLights", "shadowMapType", "alphaTest", "doubleSided", "flipSided"];
    this.getParameters = function(o, s, l, h) {
        var c = a[o.type]
          , u = i(h)
          , d = e.getPrecision();
        null !== o.precision && (d = t.getMaxPrecision(o.precision),
        d !== o.precision && console.warn("THREE.WebGLProgram.getParameters:", o.precision, "not supported, using", d, "instead."));
        var p = {
            shaderID: c,
            precision: d,
            supportsVertexTextures: t.vertexTextures,
            outputEncoding: r(e.getCurrentRenderTarget(), e.gammaOutput),
            map: !!o.map,
            mapEncoding: r(o.map, e.gammaInput),
            envMap: !!o.envMap,
            envMapMode: o.envMap && o.envMap.mapping,
            envMapEncoding: r(o.envMap, e.gammaInput),
            envMapCubeUV: !!o.envMap && (o.envMap.mapping === n.CubeUVReflectionMapping || o.envMap.mapping === n.CubeUVRefractionMapping),
            lightMap: !!o.lightMap,
            aoMap: !!o.aoMap,
            emissiveMap: !!o.emissiveMap,
            emissiveMapEncoding: r(o.emissiveMap, e.gammaInput),
            bumpMap: !!o.bumpMap,
            normalMap: !!o.normalMap,
            displacementMap: !!o.displacementMap,
            roughnessMap: !!o.roughnessMap,
            metalnessMap: !!o.metalnessMap,
            specularMap: !!o.specularMap,
            alphaMap: !!o.alphaMap,
            combine: o.combine,
            vertexColors: o.vertexColors,
            fog: l,
            useFog: o.fog,
            fogExp: l instanceof n.FogExp2,
            flatShading: o.shading === n.FlatShading,
            sizeAttenuation: o.sizeAttenuation,
            logarithmicDepthBuffer: t.logarithmicDepthBuffer,
            skinning: o.skinning,
            maxBones: u,
            useVertexTexture: t.floatVertexTextures && h && h.skeleton && h.skeleton.useVertexTexture,
            morphTargets: o.morphTargets,
            morphNormals: o.morphNormals,
            maxMorphTargets: e.maxMorphTargets,
            maxMorphNormals: e.maxMorphNormals,
            numDirLights: s.directional.length,
            numPointLights: s.point.length,
            numSpotLights: s.spot.length,
            numHemiLights: s.hemi.length,
            pointLightShadows: s.shadowsPointLight,
            shadowMapEnabled: e.shadowMap.enabled && h.receiveShadow && s.shadows.length > 0,
            shadowMapType: e.shadowMap.type,
            toneMapping: e.toneMapping,
            physicallyCorrectLights: e.physicallyCorrectLights,
            premultipliedAlpha: o.premultipliedAlpha,
            alphaTest: o.alphaTest,
            doubleSided: o.side === n.DoubleSide,
            flipSided: o.side === n.BackSide
        };
        return p
    }
    ,
    this.getProgramCode = function(e, t) {
        var i = [];
        if (t.shaderID ? i.push(t.shaderID) : (i.push(e.fragmentShader),
        i.push(e.vertexShader)),
        void 0 !== e.defines)
            for (var n in e.defines)
                i.push(n),
                i.push(e.defines[n]);
        for (var r = 0; r < s.length; r++) {
            var o = s[r];
            i.push(o),
            i.push(t[o])
        }
        return i.join()
    }
    ,
    this.acquireProgram = function(t, i, r) {
        for (var a, s = 0, l = o.length; s < l; s++) {
            var h = o[s];
            if (h.code === r) {
                a = h,
                ++a.usedTimes;
                break
            }
        }
        return void 0 === a && (a = new n.WebGLProgram(e,r,t,i),
        o.push(a)),
        a
    }
    ,
    this.releaseProgram = function(e) {
        if (0 === --e.usedTimes) {
            var t = o.indexOf(e);
            o[t] = o[o.length - 1],
            o.pop(),
            e.destroy()
        }
    }
    ,
    this.programs = o
}
,
n.WebGLProperties = function() {
    var e = {};
    this.get = function(t) {
        var i = t.uuid
          , n = e[i];
        return void 0 === n && (n = {},
        e[i] = n),
        n
    }
    ,
    this.delete = function(t) {
        delete e[t.uuid]
    }
    ,
    this.clear = function() {
        e = {}
    }
}
,
n.WebGLShader = function() {
    function e(e) {
        for (var t = e.split("\n"), i = 0; i < t.length; i++)
            t[i] = i + 1 + ": " + t[i];
        return t.join("\n")
    }
    return function(t, i, n) {
        var r = t.createShader(i);
        return t.shaderSource(r, n),
        t.compileShader(r),
        t.getShaderParameter(r, t.COMPILE_STATUS) === !1 && console.error("THREE.WebGLShader: Shader couldn't compile."),
        "" !== t.getShaderInfoLog(r) && console.warn("THREE.WebGLShader: gl.getShaderInfoLog()", i === t.VERTEX_SHADER ? "vertex" : "fragment", t.getShaderInfoLog(r), e(n)),
        r
    }
}(),
n.WebGLShadowMap = function(e, t, i) {
    function r(e, t, i, r) {
        var o = e.geometry
          , a = null
          , s = v
          , l = e.customDepthMaterial;
        if (i && (s = A,
        l = e.customDistanceMaterial),
        l)
            a = l;
        else {
            var h = void 0 !== o.morphTargets && o.morphTargets.length > 0 && t.morphTargets
              , c = e instanceof n.SkinnedMesh && t.skinning
              , u = 0;
            h && (u |= f),
            c && (u |= g),
            a = s[u]
        }
        return a.visible = t.visible,
        a.wireframe = t.wireframe,
        a.wireframeLinewidth = t.wireframeLinewidth,
        i && void 0 !== a.uniforms.lightPos && a.uniforms.lightPos.value.copy(r),
        a
    }
    function o(e, t, i) {
        if (e.visible !== !1) {
            if (e.layers.test(t.layers) && (e instanceof n.Mesh || e instanceof n.Line || e instanceof n.Points) && e.castShadow && (e.frustumCulled === !1 || l.intersectsObject(e) === !0)) {
                var r = e.material;
                r.visible === !0 && (e.modelViewMatrix.multiplyMatrices(i.matrixWorldInverse, e.matrixWorld),
                p.push(e))
            }
            for (var a = e.children, s = 0, h = a.length; s < h; s++)
                o(a[s], t, i)
        }
    }
    for (var a = e.context, s = e.state, l = new n.Frustum, h = new n.Matrix4, c = new n.Vector2, u = new n.Vector3, d = new n.Vector3, p = [], f = 1, g = 2, m = (f | g) + 1, v = new Array(m), A = new Array(m), y = [new n.Vector3(1,0,0), new n.Vector3(-1,0,0), new n.Vector3(0,0,1), new n.Vector3(0,0,-1), new n.Vector3(0,1,0), new n.Vector3(0,-1,0)], C = [new n.Vector3(0,1,0), new n.Vector3(0,1,0), new n.Vector3(0,1,0), new n.Vector3(0,1,0), new n.Vector3(0,0,1), new n.Vector3(0,0,-1)], I = [new n.Vector4, new n.Vector4, new n.Vector4, new n.Vector4, new n.Vector4, new n.Vector4], b = n.ShaderLib.depthRGBA, w = n.UniformsUtils.clone(b.uniforms), E = n.ShaderLib.distanceRGBA, x = n.UniformsUtils.clone(E.uniforms), T = 0; T !== m; ++T) {
        var M = 0 !== (T & f)
          , S = 0 !== (T & g)
          , _ = new n.ShaderMaterial({
            uniforms: w,
            vertexShader: b.vertexShader,
            fragmentShader: b.fragmentShader,
            morphTargets: M,
            skinning: S
        });
        v[T] = _;
        var P = new n.ShaderMaterial({
            defines: {
                USE_SHADOWMAP: ""
            },
            uniforms: x,
            vertexShader: E.vertexShader,
            fragmentShader: E.fragmentShader,
            morphTargets: M,
            skinning: S
        });
        A[T] = P
    }
    var R = this;
    this.enabled = !1,
    this.autoUpdate = !0,
    this.needsUpdate = !1,
    this.type = n.PCFShadowMap,
    this.cullFace = n.CullFaceFront,
    this.render = function(f, g) {
        var m, v, A = t.shadows;
        if (0 !== A.length && R.enabled !== !1 && (R.autoUpdate !== !1 || R.needsUpdate !== !1)) {
            s.clearColor(1, 1, 1, 1),
            s.disable(a.BLEND),
            s.enable(a.CULL_FACE),
            a.frontFace(a.CCW),
            a.cullFace(R.cullFace === n.CullFaceFront ? a.FRONT : a.BACK),
            s.setDepthTest(!0),
            s.setScissorTest(!1);
            for (var b = 0, w = A.length; b < w; b++) {
                var E = A[b]
                  , x = E.shadow
                  , T = x.camera;
                if (c.copy(x.mapSize),
                E instanceof n.PointLight) {
                    m = 6,
                    v = !0;
                    var M = c.x
                      , S = c.y;
                    I[0].set(2 * M, S, M, S),
                    I[1].set(0, S, M, S),
                    I[2].set(3 * M, S, M, S),
                    I[3].set(M, S, M, S),
                    I[4].set(3 * M, 0, M, S),
                    I[5].set(M, 0, M, S),
                    c.x *= 4,
                    c.y *= 2
                } else
                    m = 1,
                    v = !1;
                if (null === x.map) {
                    var _ = {
                        minFilter: n.NearestFilter,
                        magFilter: n.NearestFilter,
                        format: n.RGBAFormat
                    };
                    x.map = new n.WebGLRenderTarget(c.x,c.y,_),
                    E instanceof n.SpotLight && (T.aspect = c.x / c.y),
                    T.updateProjectionMatrix()
                }
                var P = x.map
                  , L = x.matrix;
                d.setFromMatrixPosition(E.matrixWorld),
                T.position.copy(d),
                e.setRenderTarget(P),
                e.clear();
                for (var O = 0; O < m; O++) {
                    if (v) {
                        u.copy(T.position),
                        u.add(y[O]),
                        T.up.copy(C[O]),
                        T.lookAt(u);
                        var D = I[O];
                        s.viewport(D)
                    } else
                        u.setFromMatrixPosition(E.target.matrixWorld),
                        T.lookAt(u);
                    T.updateMatrixWorld(),
                    T.matrixWorldInverse.getInverse(T.matrixWorld),
                    L.set(.5, 0, 0, .5, 0, .5, 0, .5, 0, 0, .5, .5, 0, 0, 0, 1),
                    L.multiply(T.projectionMatrix),
                    L.multiply(T.matrixWorldInverse),
                    h.multiplyMatrices(T.projectionMatrix, T.matrixWorldInverse),
                    l.setFromMatrix(h),
                    p.length = 0,
                    o(f, g, T);
                    for (var F = 0, N = p.length; F < N; F++) {
                        var B = p[F]
                          , k = i.update(B)
                          , U = B.material;
                        if (U instanceof n.MultiMaterial)
                            for (var V = k.groups, z = U.materials, G = 0, H = V.length; G < H; G++) {
                                var W = V[G]
                                  , j = z[W.materialIndex];
                                if (j.visible === !0) {
                                    var Y = r(B, j, v, d);
                                    e.renderBufferDirect(T, null, k, Y, B, W)
                                }
                            }
                        else {
                            var Y = r(B, U, v, d);
                            e.renderBufferDirect(T, null, k, Y, B, null)
                        }
                    }
                }
            }
            var X = e.getClearColor()
              , Z = e.getClearAlpha();
            e.setClearColor(X, Z),
            s.enable(a.BLEND),
            R.cullFace === n.CullFaceFront && a.cullFace(a.BACK),
            R.needsUpdate = !1
        }
    }
}
,
n.WebGLState = function(e, t, i) {
    var r = this
      , o = new n.Vector4
      , a = new Uint8Array(16)
      , s = new Uint8Array(16)
      , l = new Uint8Array(16)
      , h = {}
      , c = null
      , u = null
      , d = null
      , p = null
      , f = null
      , g = null
      , m = null
      , v = null
      , A = !1
      , y = null
      , C = null
      , I = null
      , b = null
      , w = null
      , E = null
      , x = null
      , T = null
      , M = null
      , S = null
      , _ = null
      , P = null
      , R = null
      , L = null
      , O = null
      , D = e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS)
      , F = void 0
      , N = {}
      , B = new n.Vector4
      , k = null
      , U = null
      , V = new n.Vector4
      , z = new n.Vector4
      , G = e.createTexture();
    e.bindTexture(e.TEXTURE_2D, G),
    e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.LINEAR),
    e.texImage2D(e.TEXTURE_2D, 0, e.RGB, 1, 1, 0, e.RGB, e.UNSIGNED_BYTE, new Uint8Array(3)),
    this.init = function() {
        this.clearColor(0, 0, 0, 1),
        this.clearDepth(1),
        this.clearStencil(0),
        this.enable(e.DEPTH_TEST),
        e.depthFunc(e.LEQUAL),
        e.frontFace(e.CCW),
        e.cullFace(e.BACK),
        this.enable(e.CULL_FACE),
        this.enable(e.BLEND),
        e.blendEquation(e.FUNC_ADD),
        e.blendFunc(e.SRC_ALPHA, e.ONE_MINUS_SRC_ALPHA)
    }
    ,
    this.initAttributes = function() {
        for (var e = 0, t = a.length; e < t; e++)
            a[e] = 0
    }
    ,
    this.enableAttribute = function(i) {
        if (a[i] = 1,
        0 === s[i] && (e.enableVertexAttribArray(i),
        s[i] = 1),
        0 !== l[i]) {
            var n = t.get("ANGLE_instanced_arrays");
            n.vertexAttribDivisorANGLE(i, 0),
            l[i] = 0
        }
    }
    ,
    this.enableAttributeAndDivisor = function(t, i, n) {
        a[t] = 1,
        0 === s[t] && (e.enableVertexAttribArray(t),
        s[t] = 1),
        l[t] !== i && (n.vertexAttribDivisorANGLE(t, i),
        l[t] = i)
    }
    ,
    this.disableUnusedAttributes = function() {
        for (var t = 0, i = s.length; t < i; t++)
            s[t] !== a[t] && (e.disableVertexAttribArray(t),
            s[t] = 0)
    }
    ,
    this.enable = function(t) {
        h[t] !== !0 && (e.enable(t),
        h[t] = !0)
    }
    ,
    this.disable = function(t) {
        h[t] !== !1 && (e.disable(t),
        h[t] = !1)
    }
    ,
    this.getCompressedTextureFormats = function() {
        if (null === c && (c = [],
        t.get("WEBGL_compressed_texture_pvrtc") || t.get("WEBGL_compressed_texture_s3tc") || t.get("WEBGL_compressed_texture_etc1")))
            for (var i = e.getParameter(e.COMPRESSED_TEXTURE_FORMATS), n = 0; n < i.length; n++)
                c.push(i[n]);
        return c
    }
    ,
    this.setBlending = function(t, r, o, a, s, l, h, c) {
        t === n.NoBlending ? this.disable(e.BLEND) : this.enable(e.BLEND),
        t === u && c === A || (t === n.AdditiveBlending ? c ? (e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD),
        e.blendFuncSeparate(e.ONE, e.ONE, e.ONE, e.ONE)) : (e.blendEquation(e.FUNC_ADD),
        e.blendFunc(e.SRC_ALPHA, e.ONE)) : t === n.SubtractiveBlending ? c ? (e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD),
        e.blendFuncSeparate(e.ZERO, e.ZERO, e.ONE_MINUS_SRC_COLOR, e.ONE_MINUS_SRC_ALPHA)) : (e.blendEquation(e.FUNC_ADD),
        e.blendFunc(e.ZERO, e.ONE_MINUS_SRC_COLOR)) : t === n.MultiplyBlending ? c ? (e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD),
        e.blendFuncSeparate(e.ZERO, e.ZERO, e.SRC_COLOR, e.SRC_ALPHA)) : (e.blendEquation(e.FUNC_ADD),
        e.blendFunc(e.ZERO, e.SRC_COLOR)) : c ? (e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD),
        e.blendFuncSeparate(e.ONE, e.ONE_MINUS_SRC_ALPHA, e.ONE, e.ONE_MINUS_SRC_ALPHA)) : (e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD),
        e.blendFuncSeparate(e.SRC_ALPHA, e.ONE_MINUS_SRC_ALPHA, e.ONE, e.ONE_MINUS_SRC_ALPHA)),
        u = t,
        A = c),
        t === n.CustomBlending ? (s = s || r,
        l = l || o,
        h = h || a,
        r === d && s === g || (e.blendEquationSeparate(i(r), i(s)),
        d = r,
        g = s),
        o === p && a === f && l === m && h === v || (e.blendFuncSeparate(i(o), i(a), i(l), i(h)),
        p = o,
        f = a,
        m = l,
        v = h)) : (d = null,
        p = null,
        f = null,
        g = null,
        m = null,
        v = null)
    }
    ,
    this.setDepthFunc = function(t) {
        if (y !== t) {
            if (t)
                switch (t) {
                case n.NeverDepth:
                    e.depthFunc(e.NEVER);
                    break;
                case n.AlwaysDepth:
                    e.depthFunc(e.ALWAYS);
                    break;
                case n.LessDepth:
                    e.depthFunc(e.LESS);
                    break;
                case n.LessEqualDepth:
                    e.depthFunc(e.LEQUAL);
                    break;
                case n.EqualDepth:
                    e.depthFunc(e.EQUAL);
                    break;
                case n.GreaterEqualDepth:
                    e.depthFunc(e.GEQUAL);
                    break;
                case n.GreaterDepth:
                    e.depthFunc(e.GREATER);
                    break;
                case n.NotEqualDepth:
                    e.depthFunc(e.NOTEQUAL);
                    break;
                default:
                    e.depthFunc(e.LEQUAL)
                }
            else
                e.depthFunc(e.LEQUAL);
            y = t
        }
    }
    ,
    this.setDepthTest = function(t) {
        t ? this.enable(e.DEPTH_TEST) : this.disable(e.DEPTH_TEST)
    }
    ,
    this.setDepthWrite = function(t) {
        C !== t && (e.depthMask(t),
        C = t)
    }
    ,
    this.setColorWrite = function(t) {
        I !== t && (e.colorMask(t, t, t, t),
        I = t)
    }
    ,
    this.setStencilFunc = function(t, i, n) {
        w === t && E === i && x === n || (e.stencilFunc(t, i, n),
        w = t,
        E = i,
        x = n)
    }
    ,
    this.setStencilOp = function(t, i, n) {
        T === t && M === i && S === n || (e.stencilOp(t, i, n),
        T = t,
        M = i,
        S = n)
    }
    ,
    this.setStencilTest = function(t) {
        t ? this.enable(e.STENCIL_TEST) : this.disable(e.STENCIL_TEST)
    }
    ,
    this.setStencilWrite = function(t) {
        b !== t && (e.stencilMask(t),
        b = t)
    }
    ,
    this.setFlipSided = function(t) {
        _ !== t && (t ? e.frontFace(e.CW) : e.frontFace(e.CCW),
        _ = t)
    }
    ,
    this.setLineWidth = function(t) {
        t !== P && (e.lineWidth(t),
        P = t)
    }
    ,
    this.setPolygonOffset = function(t, i, n) {
        t ? this.enable(e.POLYGON_OFFSET_FILL) : this.disable(e.POLYGON_OFFSET_FILL),
        !t || R === i && L === n || (e.polygonOffset(i, n),
        R = i,
        L = n)
    }
    ,
    this.getScissorTest = function() {
        return O
    }
    ,
    this.setScissorTest = function(t) {
        O = t,
        t ? this.enable(e.SCISSOR_TEST) : this.disable(e.SCISSOR_TEST)
    }
    ,
    this.activeTexture = function(t) {
        void 0 === t && (t = e.TEXTURE0 + D - 1),
        F !== t && (e.activeTexture(t),
        F = t)
    }
    ,
    this.bindTexture = function(t, i) {
        void 0 === F && r.activeTexture();
        var n = N[F];
        void 0 === n && (n = {
            type: void 0,
            texture: void 0
        },
        N[F] = n),
        n.type === t && n.texture === i || (e.bindTexture(t, i || G),
        n.type = t,
        n.texture = i)
    }
    ,
    this.compressedTexImage2D = function() {
        try {
            e.compressedTexImage2D.apply(e, arguments)
        } catch (e) {
            console.error(e)
        }
    }
    ,
    this.texImage2D = function() {
        try {
            e.texImage2D.apply(e, arguments)
        } catch (e) {
            console.error(e)
        }
    }
    ,
    this.clearColor = function(t, i, n, r) {
        o.set(t, i, n, r),
        B.equals(o) === !1 && (e.clearColor(t, i, n, r),
        B.copy(o))
    }
    ,
    this.clearDepth = function(t) {
        k !== t && (e.clearDepth(t),
        k = t)
    }
    ,
    this.clearStencil = function(t) {
        U !== t && (e.clearStencil(t),
        U = t)
    }
    ,
    this.scissor = function(t) {
        V.equals(t) === !1 && (e.scissor(t.x, t.y, t.z, t.w),
        V.copy(t))
    }
    ,
    this.viewport = function(t) {
        z.equals(t) === !1 && (e.viewport(t.x, t.y, t.z, t.w),
        z.copy(t))
    }
    ,
    this.reset = function() {
        for (var t = 0; t < s.length; t++)
            1 === s[t] && (e.disableVertexAttribArray(t),
            s[t] = 0);
        h = {},
        c = null,
        F = void 0,
        N = {},
        u = null,
        I = null,
        C = null,
        b = null,
        _ = null
    }
}
,
n.LensFlarePlugin = function(e, t) {
    function i() {
        var e = new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1])
          , t = new Uint16Array([0, 1, 2, 0, 2, 3]);
        o = p.createBuffer(),
        a = p.createBuffer(),
        p.bindBuffer(p.ARRAY_BUFFER, o),
        p.bufferData(p.ARRAY_BUFFER, e, p.STATIC_DRAW),
        p.bindBuffer(p.ELEMENT_ARRAY_BUFFER, a),
        p.bufferData(p.ELEMENT_ARRAY_BUFFER, t, p.STATIC_DRAW),
        u = p.createTexture(),
        d = p.createTexture(),
        f.bindTexture(p.TEXTURE_2D, u),
        p.texImage2D(p.TEXTURE_2D, 0, p.RGB, 16, 16, 0, p.RGB, p.UNSIGNED_BYTE, null),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_WRAP_S, p.CLAMP_TO_EDGE),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_WRAP_T, p.CLAMP_TO_EDGE),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_MAG_FILTER, p.NEAREST),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_MIN_FILTER, p.NEAREST),
        f.bindTexture(p.TEXTURE_2D, d),
        p.texImage2D(p.TEXTURE_2D, 0, p.RGBA, 16, 16, 0, p.RGBA, p.UNSIGNED_BYTE, null),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_WRAP_S, p.CLAMP_TO_EDGE),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_WRAP_T, p.CLAMP_TO_EDGE),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_MAG_FILTER, p.NEAREST),
        p.texParameteri(p.TEXTURE_2D, p.TEXTURE_MIN_FILTER, p.NEAREST),
        c = p.getParameter(p.MAX_VERTEX_TEXTURE_IMAGE_UNITS) > 0;
        var i;
        i = c ? {
            vertexShader: ["uniform lowp int renderType;", "uniform vec3 screenPosition;", "uniform vec2 scale;", "uniform float rotation;", "uniform sampler2D occlusionMap;", "attribute vec2 position;", "attribute vec2 uv;", "varying vec2 vUV;", "varying float vVisibility;", "void main() {", "vUV = uv;", "vec2 pos = position;", "if ( renderType == 2 ) {", "vec4 visibility = texture2D( occlusionMap, vec2( 0.1, 0.1 ) );", "visibility += texture2D( occlusionMap, vec2( 0.5, 0.1 ) );", "visibility += texture2D( occlusionMap, vec2( 0.9, 0.1 ) );", "visibility += texture2D( occlusionMap, vec2( 0.9, 0.5 ) );", "visibility += texture2D( occlusionMap, vec2( 0.9, 0.9 ) );", "visibility += texture2D( occlusionMap, vec2( 0.5, 0.9 ) );", "visibility += texture2D( occlusionMap, vec2( 0.1, 0.9 ) );", "visibility += texture2D( occlusionMap, vec2( 0.1, 0.5 ) );", "visibility += texture2D( occlusionMap, vec2( 0.5, 0.5 ) );", "vVisibility =        visibility.r / 9.0;", "vVisibility *= 1.0 - visibility.g / 9.0;", "vVisibility *=       visibility.b / 9.0;", "vVisibility *= 1.0 - visibility.a / 9.0;", "pos.x = cos( rotation ) * position.x - sin( rotation ) * position.y;", "pos.y = sin( rotation ) * position.x + cos( rotation ) * position.y;", "}", "gl_Position = vec4( ( pos * scale + screenPosition.xy ).xy, screenPosition.z, 1.0 );", "}"].join("\n"),
            fragmentShader: ["uniform lowp int renderType;", "uniform sampler2D map;", "uniform float opacity;", "uniform vec3 color;", "varying vec2 vUV;", "varying float vVisibility;", "void main() {", "if ( renderType == 0 ) {", "gl_FragColor = vec4( 1.0, 0.0, 1.0, 0.0 );", "} else if ( renderType == 1 ) {", "gl_FragColor = texture2D( map, vUV );", "} else {", "vec4 texture = texture2D( map, vUV );", "texture.a *= opacity * vVisibility;", "gl_FragColor = texture;", "gl_FragColor.rgb *= color;", "}", "}"].join("\n")
        } : {
            vertexShader: ["uniform lowp int renderType;", "uniform vec3 screenPosition;", "uniform vec2 scale;", "uniform float rotation;", "attribute vec2 position;", "attribute vec2 uv;", "varying vec2 vUV;", "void main() {", "vUV = uv;", "vec2 pos = position;", "if ( renderType == 2 ) {", "pos.x = cos( rotation ) * position.x - sin( rotation ) * position.y;", "pos.y = sin( rotation ) * position.x + cos( rotation ) * position.y;", "}", "gl_Position = vec4( ( pos * scale + screenPosition.xy ).xy, screenPosition.z, 1.0 );", "}"].join("\n"),
            fragmentShader: ["precision mediump float;", "uniform lowp int renderType;", "uniform sampler2D map;", "uniform sampler2D occlusionMap;", "uniform float opacity;", "uniform vec3 color;", "varying vec2 vUV;", "void main() {", "if ( renderType == 0 ) {", "gl_FragColor = vec4( texture2D( map, vUV ).rgb, 0.0 );", "} else if ( renderType == 1 ) {", "gl_FragColor = texture2D( map, vUV );", "} else {", "float visibility = texture2D( occlusionMap, vec2( 0.5, 0.1 ) ).a;", "visibility += texture2D( occlusionMap, vec2( 0.9, 0.5 ) ).a;", "visibility += texture2D( occlusionMap, vec2( 0.5, 0.9 ) ).a;", "visibility += texture2D( occlusionMap, vec2( 0.1, 0.5 ) ).a;", "visibility = ( 1.0 - visibility / 4.0 );", "vec4 texture = texture2D( map, vUV );", "texture.a *= opacity * visibility;", "gl_FragColor = texture;", "gl_FragColor.rgb *= color;", "}", "}"].join("\n")
        },
        s = r(i),
        l = {
            vertex: p.getAttribLocation(s, "position"),
            uv: p.getAttribLocation(s, "uv")
        },
        h = {
            renderType: p.getUniformLocation(s, "renderType"),
            map: p.getUniformLocation(s, "map"),
            occlusionMap: p.getUniformLocation(s, "occlusionMap"),
            opacity: p.getUniformLocation(s, "opacity"),
            color: p.getUniformLocation(s, "color"),
            scale: p.getUniformLocation(s, "scale"),
            rotation: p.getUniformLocation(s, "rotation"),
            screenPosition: p.getUniformLocation(s, "screenPosition")
        }
    }
    function r(t) {
        var i = p.createProgram()
          , n = p.createShader(p.FRAGMENT_SHADER)
          , r = p.createShader(p.VERTEX_SHADER)
          , o = "precision " + e.getPrecision() + " float;\n";
        return p.shaderSource(n, o + t.fragmentShader),
        p.shaderSource(r, o + t.vertexShader),
        p.compileShader(n),
        p.compileShader(r),
        p.attachShader(i, n),
        p.attachShader(i, r),
        p.linkProgram(i),
        i
    }
    var o, a, s, l, h, c, u, d, p = e.context, f = e.state;
    this.render = function(r, g, m) {
        if (0 !== t.length) {
            var v = new n.Vector3
              , A = m.w / m.z
              , y = .5 * m.z
              , C = .5 * m.w
              , I = 16 / m.w
              , b = new n.Vector2(I * A,I)
              , w = new n.Vector3(1,1,0)
              , E = new n.Vector2(1,1);
            void 0 === s && i(),
            p.useProgram(s),
            f.initAttributes(),
            f.enableAttribute(l.vertex),
            f.enableAttribute(l.uv),
            f.disableUnusedAttributes(),
            p.uniform1i(h.occlusionMap, 0),
            p.uniform1i(h.map, 1),
            p.bindBuffer(p.ARRAY_BUFFER, o),
            p.vertexAttribPointer(l.vertex, 2, p.FLOAT, !1, 16, 0),
            p.vertexAttribPointer(l.uv, 2, p.FLOAT, !1, 16, 8),
            p.bindBuffer(p.ELEMENT_ARRAY_BUFFER, a),
            f.disable(p.CULL_FACE),
            f.setDepthWrite(!1);
            for (var x = 0, T = t.length; x < T; x++) {
                I = 16 / m.w,
                b.set(I * A, I);
                var M = t[x];
                if (v.set(M.matrixWorld.elements[12], M.matrixWorld.elements[13], M.matrixWorld.elements[14]),
                v.applyMatrix4(g.matrixWorldInverse),
                v.applyProjection(g.projectionMatrix),
                w.copy(v),
                E.x = w.x * y + y,
                E.y = w.y * C + C,
                c || E.x > 0 && E.x < m.z && E.y > 0 && E.y < m.w) {
                    f.activeTexture(p.TEXTURE0),
                    f.bindTexture(p.TEXTURE_2D, null),
                    f.activeTexture(p.TEXTURE1),
                    f.bindTexture(p.TEXTURE_2D, u),
                    p.copyTexImage2D(p.TEXTURE_2D, 0, p.RGB, m.x + E.x - 8, m.y + E.y - 8, 16, 16, 0),
                    p.uniform1i(h.renderType, 0),
                    p.uniform2f(h.scale, b.x, b.y),
                    p.uniform3f(h.screenPosition, w.x, w.y, w.z),
                    f.disable(p.BLEND),
                    f.enable(p.DEPTH_TEST),
                    p.drawElements(p.TRIANGLES, 6, p.UNSIGNED_SHORT, 0),
                    f.activeTexture(p.TEXTURE0),
                    f.bindTexture(p.TEXTURE_2D, d),
                    p.copyTexImage2D(p.TEXTURE_2D, 0, p.RGBA, m.x + E.x - 8, m.y + E.y - 8, 16, 16, 0),
                    p.uniform1i(h.renderType, 1),
                    f.disable(p.DEPTH_TEST),
                    f.activeTexture(p.TEXTURE1),
                    f.bindTexture(p.TEXTURE_2D, u),
                    p.drawElements(p.TRIANGLES, 6, p.UNSIGNED_SHORT, 0),
                    M.positionScreen.copy(w),
                    M.customUpdateCallback ? M.customUpdateCallback(M) : M.updateLensFlares(),
                    p.uniform1i(h.renderType, 2),
                    f.enable(p.BLEND);
                    for (var S = 0, _ = M.lensFlares.length; S < _; S++) {
                        var P = M.lensFlares[S];
                        P.opacity > .001 && P.scale > .001 && (w.x = P.x,
                        w.y = P.y,
                        w.z = P.z,
                        I = P.size * P.scale / m.w,
                        b.x = I * A,
                        b.y = I,
                        p.uniform3f(h.screenPosition, w.x, w.y, w.z),
                        p.uniform2f(h.scale, b.x, b.y),
                        p.uniform1f(h.rotation, P.rotation),
                        p.uniform1f(h.opacity, P.opacity),
                        p.uniform3f(h.color, P.color.r, P.color.g, P.color.b),
                        f.setBlending(P.blending, P.blendEquation, P.blendSrc, P.blendDst),
                        e.setTexture(P.texture, 1),
                        p.drawElements(p.TRIANGLES, 6, p.UNSIGNED_SHORT, 0))
                    }
                }
            }
            f.enable(p.CULL_FACE),
            f.enable(p.DEPTH_TEST),
            f.setDepthWrite(!0),
            e.resetGLState()
        }
    }
}
,
n.SpritePlugin = function(e, t) {
    function i() {
        var e = new Float32Array([-.5, -.5, 0, 0, .5, -.5, 1, 0, .5, .5, 1, 1, -.5, .5, 0, 1])
          , t = new Uint16Array([0, 1, 2, 0, 2, 3]);
        a = d.createBuffer(),
        s = d.createBuffer(),
        d.bindBuffer(d.ARRAY_BUFFER, a),
        d.bufferData(d.ARRAY_BUFFER, e, d.STATIC_DRAW),
        d.bindBuffer(d.ELEMENT_ARRAY_BUFFER, s),
        d.bufferData(d.ELEMENT_ARRAY_BUFFER, t, d.STATIC_DRAW),
        l = r(),
        h = {
            position: d.getAttribLocation(l, "position"),
            uv: d.getAttribLocation(l, "uv")
        },
        c = {
            uvOffset: d.getUniformLocation(l, "uvOffset"),
            uvScale: d.getUniformLocation(l, "uvScale"),
            rotation: d.getUniformLocation(l, "rotation"),
            scale: d.getUniformLocation(l, "scale"),
            color: d.getUniformLocation(l, "color"),
            map: d.getUniformLocation(l, "map"),
            opacity: d.getUniformLocation(l, "opacity"),
            modelViewMatrix: d.getUniformLocation(l, "modelViewMatrix"),
            projectionMatrix: d.getUniformLocation(l, "projectionMatrix"),
            fogType: d.getUniformLocation(l, "fogType"),
            fogDensity: d.getUniformLocation(l, "fogDensity"),
            fogNear: d.getUniformLocation(l, "fogNear"),
            fogFar: d.getUniformLocation(l, "fogFar"),
            fogColor: d.getUniformLocation(l, "fogColor"),
            alphaTest: d.getUniformLocation(l, "alphaTest")
        };
        var i = document.createElement("canvas");
        i.width = 8,
        i.height = 8;
        var o = i.getContext("2d");
        o.fillStyle = "white",
        o.fillRect(0, 0, 8, 8),
        u = new n.Texture(i),
        u.needsUpdate = !0
    }
    function r() {
        var t = d.createProgram()
          , i = d.createShader(d.VERTEX_SHADER)
          , n = d.createShader(d.FRAGMENT_SHADER);
        return d.shaderSource(i, ["precision " + e.getPrecision() + " float;", "uniform mat4 modelViewMatrix;", "uniform mat4 projectionMatrix;", "uniform float rotation;", "uniform vec2 scale;", "uniform vec2 uvOffset;", "uniform vec2 uvScale;", "attribute vec2 position;", "attribute vec2 uv;", "varying vec2 vUV;", "void main() {", "vUV = uvOffset + uv * uvScale;", "vec2 alignedPosition = position * scale;", "vec2 rotatedPosition;", "rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;", "rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;", "vec4 finalPosition;", "finalPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );", "finalPosition.xy += rotatedPosition;", "finalPosition = projectionMatrix * finalPosition;", "gl_Position = finalPosition;", "}"].join("\n")),
        d.shaderSource(n, ["precision " + e.getPrecision() + " float;", "uniform vec3 color;", "uniform sampler2D map;", "uniform float opacity;", "uniform int fogType;", "uniform vec3 fogColor;", "uniform float fogDensity;", "uniform float fogNear;", "uniform float fogFar;", "uniform float alphaTest;", "varying vec2 vUV;", "void main() {", "vec4 texture = texture2D( map, vUV );", "if ( texture.a < alphaTest ) discard;", "gl_FragColor = vec4( color * texture.xyz, texture.a * opacity );", "if ( fogType > 0 ) {", "float depth = gl_FragCoord.z / gl_FragCoord.w;", "float fogFactor = 0.0;", "if ( fogType == 1 ) {", "fogFactor = smoothstep( fogNear, fogFar, depth );", "} else {", "const float LOG2 = 1.442695;", "fogFactor = exp2( - fogDensity * fogDensity * depth * depth * LOG2 );", "fogFactor = 1.0 - clamp( fogFactor, 0.0, 1.0 );", "}", "gl_FragColor = mix( gl_FragColor, vec4( fogColor, gl_FragColor.w ), fogFactor );", "}", "}"].join("\n")),
        d.compileShader(i),
        d.compileShader(n),
        d.attachShader(t, i),
        d.attachShader(t, n),
        d.linkProgram(t),
        t
    }
    function o(e, t) {
        return e.renderOrder !== t.renderOrder ? e.renderOrder - t.renderOrder : e.z !== t.z ? t.z - e.z : t.id - e.id
    }
    var a, s, l, h, c, u, d = e.context, p = e.state, f = new n.Vector3, g = new n.Quaternion, m = new n.Vector3;
    this.render = function(r, v) {
        if (0 !== t.length) {
            void 0 === l && i(),
            d.useProgram(l),
            p.initAttributes(),
            p.enableAttribute(h.position),
            p.enableAttribute(h.uv),
            p.disableUnusedAttributes(),
            p.disable(d.CULL_FACE),
            p.enable(d.BLEND),
            d.bindBuffer(d.ARRAY_BUFFER, a),
            d.vertexAttribPointer(h.position, 2, d.FLOAT, !1, 16, 0),
            d.vertexAttribPointer(h.uv, 2, d.FLOAT, !1, 16, 8),
            d.bindBuffer(d.ELEMENT_ARRAY_BUFFER, s),
            d.uniformMatrix4fv(c.projectionMatrix, !1, v.projectionMatrix.elements),
            p.activeTexture(d.TEXTURE0),
            d.uniform1i(c.map, 0);
            var A = 0
              , y = 0
              , C = r.fog;
            C ? (d.uniform3f(c.fogColor, C.color.r, C.color.g, C.color.b),
            C instanceof n.Fog ? (d.uniform1f(c.fogNear, C.near),
            d.uniform1f(c.fogFar, C.far),
            d.uniform1i(c.fogType, 1),
            A = 1,
            y = 1) : C instanceof n.FogExp2 && (d.uniform1f(c.fogDensity, C.density),
            d.uniform1i(c.fogType, 2),
            A = 2,
            y = 2)) : (d.uniform1i(c.fogType, 0),
            A = 0,
            y = 0);
            for (var I = 0, b = t.length; I < b; I++) {
                var w = t[I];
                w.modelViewMatrix.multiplyMatrices(v.matrixWorldInverse, w.matrixWorld),
                w.z = -w.modelViewMatrix.elements[14]
            }
            t.sort(o);
            for (var E = [], I = 0, b = t.length; I < b; I++) {
                var w = t[I]
                  , x = w.material;
                d.uniform1f(c.alphaTest, x.alphaTest),
                d.uniformMatrix4fv(c.modelViewMatrix, !1, w.modelViewMatrix.elements),
                w.matrixWorld.decompose(f, g, m),
                E[0] = m.x,
                E[1] = m.y;
                var T = 0;
                r.fog && x.fog && (T = y),
                A !== T && (d.uniform1i(c.fogType, T),
                A = T),
                null !== x.map ? (d.uniform2f(c.uvOffset, x.map.offset.x, x.map.offset.y),
                d.uniform2f(c.uvScale, x.map.repeat.x, x.map.repeat.y)) : (d.uniform2f(c.uvOffset, 0, 0),
                d.uniform2f(c.uvScale, 1, 1)),
                d.uniform1f(c.opacity, x.opacity),
                d.uniform3f(c.color, x.color.r, x.color.g, x.color.b),
                d.uniform1f(c.rotation, x.rotation),
                d.uniform2fv(c.scale, E),
                p.setBlending(x.blending, x.blendEquation, x.blendSrc, x.blendDst),
                p.setDepthTest(x.depthTest),
                p.setDepthWrite(x.depthWrite),
                x.map && x.map.image && x.map.image.width ? e.setTexture(x.map, 0) : e.setTexture(u, 0),
                d.drawElements(d.TRIANGLES, 6, d.UNSIGNED_SHORT, 0)
            }
            p.enable(d.CULL_FACE),
            e.resetGLState()
        }
    }
}
,
Object.defineProperties(n.Box2.prototype, {
    empty: {
        value: function() {
            return console.warn("THREE.Box2: .empty() has been renamed to .isEmpty()."),
            this.isEmpty()
        }
    },
    isIntersectionBox: {
        value: function(e) {
            return console.warn("THREE.Box2: .isIntersectionBox() has been renamed to .intersectsBox()."),
            this.intersectsBox(e)
        }
    }
}),
Object.defineProperties(n.Box3.prototype, {
    empty: {
        value: function() {
            return console.warn("THREE.Box3: .empty() has been renamed to .isEmpty()."),
            this.isEmpty()
        }
    },
    isIntersectionBox: {
        value: function(e) {
            return console.warn("THREE.Box3: .isIntersectionBox() has been renamed to .intersectsBox()."),
            this.intersectsBox(e)
        }
    },
    isIntersectionSphere: {
        value: function(e) {
            return console.warn("THREE.Box3: .isIntersectionSphere() has been renamed to .intersectsSphere()."),
            this.intersectsSphere(e)
        }
    }
}),
Object.defineProperties(n.Matrix3.prototype, {
    multiplyVector3: {
        value: function(e) {
            return console.warn("THREE.Matrix3: .multiplyVector3() has been removed. Use vector.applyMatrix3( matrix ) instead."),
            e.applyMatrix3(this)
        }
    },
    multiplyVector3Array: {
        value: function(e) {
            return console.warn("THREE.Matrix3: .multiplyVector3Array() has been renamed. Use matrix.applyToVector3Array( array ) instead."),
            this.applyToVector3Array(e)
        }
    }
}),
Object.defineProperties(n.Matrix4.prototype, {
    extractPosition: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .extractPosition() has been renamed to .copyPosition()."),
            this.copyPosition(e)
        }
    },
    setRotationFromQuaternion: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .setRotationFromQuaternion() has been renamed to .makeRotationFromQuaternion()."),
            this.makeRotationFromQuaternion(e)
        }
    },
    multiplyVector3: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .multiplyVector3() has been removed. Use vector.applyMatrix4( matrix ) or vector.applyProjection( matrix ) instead."),
            e.applyProjection(this)
        }
    },
    multiplyVector4: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .multiplyVector4() has been removed. Use vector.applyMatrix4( matrix ) instead."),
            e.applyMatrix4(this)
        }
    },
    multiplyVector3Array: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .multiplyVector3Array() has been renamed. Use matrix.applyToVector3Array( array ) instead."),
            this.applyToVector3Array(e)
        }
    },
    rotateAxis: {
        value: function(e) {
            console.warn("THREE.Matrix4: .rotateAxis() has been removed. Use Vector3.transformDirection( matrix ) instead."),
            e.transformDirection(this)
        }
    },
    crossVector: {
        value: function(e) {
            return console.warn("THREE.Matrix4: .crossVector() has been removed. Use vector.applyMatrix4( matrix ) instead."),
            e.applyMatrix4(this)
        }
    },
    translate: {
        value: function(e) {
            console.error("THREE.Matrix4: .translate() has been removed.")
        }
    },
    rotateX: {
        value: function(e) {
            console.error("THREE.Matrix4: .rotateX() has been removed.")
        }
    },
    rotateY: {
        value: function(e) {
            console.error("THREE.Matrix4: .rotateY() has been removed.")
        }
    },
    rotateZ: {
        value: function(e) {
            console.error("THREE.Matrix4: .rotateZ() has been removed.")
        }
    },
    rotateByAxis: {
        value: function(e, t) {
            console.error("THREE.Matrix4: .rotateByAxis() has been removed.")
        }
    }
}),
Object.defineProperties(n.Plane.prototype, {
    isIntersectionLine: {
        value: function(e) {
            return console.warn("THREE.Plane: .isIntersectionLine() has been renamed to .intersectsLine()."),
            this.intersectsLine(e)
        }
    }
}),
Object.defineProperties(n.Quaternion.prototype, {
    multiplyVector3: {
        value: function(e) {
            return console.warn("THREE.Quaternion: .multiplyVector3() has been removed. Use is now vector.applyQuaternion( quaternion ) instead."),
            e.applyQuaternion(this)
        }
    }
}),
Object.defineProperties(n.Ray.prototype, {
    isIntersectionBox: {
        value: function(e) {
            return console.warn("THREE.Ray: .isIntersectionBox() has been renamed to .intersectsBox()."),
            this.intersectsBox(e)
        }
    },
    isIntersectionPlane: {
        value: function(e) {
            return console.warn("THREE.Ray: .isIntersectionPlane() has been renamed to .intersectsPlane()."),
            this.intersectsPlane(e)
        }
    },
    isIntersectionSphere: {
        value: function(e) {
            return console.warn("THREE.Ray: .isIntersectionSphere() has been renamed to .intersectsSphere()."),
            this.intersectsSphere(e)
        }
    }
}),
Object.defineProperties(n.Vector3.prototype, {
    setEulerFromRotationMatrix: {
        value: function() {
            console.error("THREE.Vector3: .setEulerFromRotationMatrix() has been removed. Use Euler.setFromRotationMatrix() instead.")
        }
    },
    setEulerFromQuaternion: {
        value: function() {
            console.error("THREE.Vector3: .setEulerFromQuaternion() has been removed. Use Euler.setFromQuaternion() instead.")
        }
    },
    getPositionFromMatrix: {
        value: function(e) {
            return console.warn("THREE.Vector3: .getPositionFromMatrix() has been renamed to .setFromMatrixPosition()."),
            this.setFromMatrixPosition(e)
        }
    },
    getScaleFromMatrix: {
        value: function(e) {
            return console.warn("THREE.Vector3: .getScaleFromMatrix() has been renamed to .setFromMatrixScale()."),
            this.setFromMatrixScale(e)
        }
    },
    getColumnFromMatrix: {
        value: function(e, t) {
            return console.warn("THREE.Vector3: .getColumnFromMatrix() has been renamed to .setFromMatrixColumn()."),
            this.setFromMatrixColumn(e, t)
        }
    }
}),
n.Face4 = function(e, t, i, r, o, a, s) {
    return console.warn("THREE.Face4 has been removed. A THREE.Face3 will be created instead."),
    new n.Face3(e,t,i,o,a,s)
}
,
n.Vertex = function(e, t, i) {
    return console.warn("THREE.Vertex has been removed. Use THREE.Vector3 instead."),
    new n.Vector3(e,t,i)
}
,
Object.defineProperties(n.Object3D.prototype, {
    eulerOrder: {
        get: function() {
            return console.warn("THREE.Object3D: .eulerOrder is now .rotation.order."),
            this.rotation.order
        },
        set: function(e) {
            console.warn("THREE.Object3D: .eulerOrder is now .rotation.order."),
            this.rotation.order = e
        }
    },
    getChildByName: {
        value: function(e) {
            return console.warn("THREE.Object3D: .getChildByName() has been renamed to .getObjectByName()."),
            this.getObjectByName(e)
        }
    },
    renderDepth: {
        set: function(e) {
            console.warn("THREE.Object3D: .renderDepth has been removed. Use .renderOrder, instead.")
        }
    },
    translate: {
        value: function(e, t) {
            return console.warn("THREE.Object3D: .translate() has been removed. Use .translateOnAxis( axis, distance ) instead."),
            this.translateOnAxis(t, e)
        }
    },
    useQuaternion: {
        get: function() {
            console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")
        },
        set: function(e) {
            console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")
        }
    }
}),
Object.defineProperties(n, {
    PointCloud: {
        value: function(e, t) {
            return console.warn("THREE.PointCloud has been renamed to THREE.Points."),
            new n.Points(e,t)
        }
    },
    ParticleSystem: {
        value: function(e, t) {
            return console.warn("THREE.ParticleSystem has been renamed to THREE.Points."),
            new n.Points(e,t)
        }
    }
}),
Object.defineProperties(n.Light.prototype, {
    onlyShadow: {
        set: function(e) {
            console.warn("THREE.Light: .onlyShadow has been removed.")
        }
    },
    shadowCameraFov: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraFov is now .shadow.camera.fov."),
            this.shadow.camera.fov = e
        }
    },
    shadowCameraLeft: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraLeft is now .shadow.camera.left."),
            this.shadow.camera.left = e
        }
    },
    shadowCameraRight: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraRight is now .shadow.camera.right."),
            this.shadow.camera.right = e
        }
    },
    shadowCameraTop: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraTop is now .shadow.camera.top."),
            this.shadow.camera.top = e
        }
    },
    shadowCameraBottom: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraBottom is now .shadow.camera.bottom."),
            this.shadow.camera.bottom = e
        }
    },
    shadowCameraNear: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraNear is now .shadow.camera.near."),
            this.shadow.camera.near = e
        }
    },
    shadowCameraFar: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraFar is now .shadow.camera.far."),
            this.shadow.camera.far = e
        }
    },
    shadowCameraVisible: {
        set: function(e) {
            console.warn("THREE.Light: .shadowCameraVisible has been removed. Use new THREE.CameraHelper( light.shadow.camera ) instead.")
        }
    },
    shadowBias: {
        set: function(e) {
            console.warn("THREE.Light: .shadowBias is now .shadow.bias."),
            this.shadow.bias = e
        }
    },
    shadowDarkness: {
        set: function(e) {
            console.warn("THREE.Light: .shadowDarkness has been removed.")
        }
    },
    shadowMapWidth: {
        set: function(e) {
            console.warn("THREE.Light: .shadowMapWidth is now .shadow.mapSize.width."),
            this.shadow.mapSize.width = e
        }
    },
    shadowMapHeight: {
        set: function(e) {
            console.warn("THREE.Light: .shadowMapHeight is now .shadow.mapSize.height."),
            this.shadow.mapSize.height = e
        }
    }
}),
Object.defineProperties(n.BufferAttribute.prototype, {
    length: {
        get: function() {
            return console.warn("THREE.BufferAttribute: .length has been deprecated. Please use .count."),
            this.array.length
        }
    }
}),
Object.defineProperties(n.BufferGeometry.prototype, {
    drawcalls: {
        get: function() {
            return console.error("THREE.BufferGeometry: .drawcalls has been renamed to .groups."),
            this.groups
        }
    },
    offsets: {
        get: function() {
            return console.warn("THREE.BufferGeometry: .offsets has been renamed to .groups."),
            this.groups
        }
    },
    addIndex: {
        value: function(e) {
            console.warn("THREE.BufferGeometry: .addIndex() has been renamed to .setIndex()."),
            this.setIndex(e)
        }
    },
    addDrawCall: {
        value: function(e, t, i) {
            void 0 !== i && console.warn("THREE.BufferGeometry: .addDrawCall() no longer supports indexOffset."),
            console.warn("THREE.BufferGeometry: .addDrawCall() is now .addGroup()."),
            this.addGroup(e, t)
        }
    },
    clearDrawCalls: {
        value: function() {
            console.warn("THREE.BufferGeometry: .clearDrawCalls() is now .clearGroups()."),
            this.clearGroups()
        }
    },
    computeTangents: {
        value: function() {
            console.warn("THREE.BufferGeometry: .computeTangents() has been removed.")
        }
    },
    computeOffsets: {
        value: function() {
            console.warn("THREE.BufferGeometry: .computeOffsets() has been removed.")
        }
    }
}),
Object.defineProperties(n.Material.prototype, {
    wrapAround: {
        get: function() {
            console.warn("THREE." + this.type + ": .wrapAround has been removed.")
        },
        set: function(e) {
            console.warn("THREE." + this.type + ": .wrapAround has been removed.")
        }
    },
    wrapRGB: {
        get: function() {
            return console.warn("THREE." + this.type + ": .wrapRGB has been removed."),
            new n.Color
        }
    }
}),
Object.defineProperties(n, {
    PointCloudMaterial: {
        value: function(e) {
            return console.warn("THREE.PointCloudMaterial has been renamed to THREE.PointsMaterial."),
            new n.PointsMaterial(e)
        }
    },
    ParticleBasicMaterial: {
        value: function(e) {
            return console.warn("THREE.ParticleBasicMaterial has been renamed to THREE.PointsMaterial."),
            new n.PointsMaterial(e)
        }
    },
    ParticleSystemMaterial: {
        value: function(e) {
            return console.warn("THREE.ParticleSystemMaterial has been renamed to THREE.PointsMaterial."),
            new n.PointsMaterial(e)
        }
    }
}),
Object.defineProperties(n.MeshPhongMaterial.prototype, {
    metal: {
        get: function() {
            return console.warn("THREE.MeshPhongMaterial: .metal has been removed. Use THREE.MeshStandardMaterial instead."),
            !1
        },
        set: function(e) {
            console.warn("THREE.MeshPhongMaterial: .metal has been removed. Use THREE.MeshStandardMaterial instead")
        }
    }
}),
Object.defineProperties(n.ShaderMaterial.prototype, {
    derivatives: {
        get: function() {
            return console.warn("THREE.ShaderMaterial: .derivatives has been moved to .extensions.derivatives."),
            this.extensions.derivatives
        },
        set: function(e) {
            console.warn("THREE. ShaderMaterial: .derivatives has been moved to .extensions.derivatives."),
            this.extensions.derivatives = e
        }
    }
}),
Object.defineProperties(n.WebGLRenderer.prototype, {
    supportsFloatTextures: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsFloatTextures() is now .extensions.get( 'OES_texture_float' )."),
            this.extensions.get("OES_texture_float")
        }
    },
    supportsHalfFloatTextures: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsHalfFloatTextures() is now .extensions.get( 'OES_texture_half_float' )."),
            this.extensions.get("OES_texture_half_float")
        }
    },
    supportsStandardDerivatives: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsStandardDerivatives() is now .extensions.get( 'OES_standard_derivatives' )."),
            this.extensions.get("OES_standard_derivatives")
        }
    },
    supportsCompressedTextureS3TC: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsCompressedTextureS3TC() is now .extensions.get( 'WEBGL_compressed_texture_s3tc' )."),
            this.extensions.get("WEBGL_compressed_texture_s3tc")
        }
    },
    supportsCompressedTexturePVRTC: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsCompressedTexturePVRTC() is now .extensions.get( 'WEBGL_compressed_texture_pvrtc' )."),
            this.extensions.get("WEBGL_compressed_texture_pvrtc")
        }
    },
    supportsBlendMinMax: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsBlendMinMax() is now .extensions.get( 'EXT_blend_minmax' )."),
            this.extensions.get("EXT_blend_minmax")
        }
    },
    supportsVertexTextures: {
        value: function() {
            return this.capabilities.vertexTextures
        }
    },
    supportsInstancedArrays: {
        value: function() {
            return console.warn("THREE.WebGLRenderer: .supportsInstancedArrays() is now .extensions.get( 'ANGLE_instanced_arrays' )."),
            this.extensions.get("ANGLE_instanced_arrays")
        }
    },
    enableScissorTest: {
        value: function(e) {
            console.warn("THREE.WebGLRenderer: .enableScissorTest() is now .setScissorTest()."),
            this.setScissorTest(e)
        }
    },
    initMaterial: {
        value: function() {
            console.warn("THREE.WebGLRenderer: .initMaterial() has been removed.")
        }
    },
    addPrePlugin: {
        value: function() {
            console.warn("THREE.WebGLRenderer: .addPrePlugin() has been removed.")
        }
    },
    addPostPlugin: {
        value: function() {
            console.warn("THREE.WebGLRenderer: .addPostPlugin() has been removed.")
        }
    },
    updateShadowMap: {
        value: function() {
            console.warn("THREE.WebGLRenderer: .updateShadowMap() has been removed.")
        }
    },
    shadowMapEnabled: {
        get: function() {
            return this.shadowMap.enabled
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderer: .shadowMapEnabled is now .shadowMap.enabled."),
            this.shadowMap.enabled = e
        }
    },
    shadowMapType: {
        get: function() {
            return this.shadowMap.type
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderer: .shadowMapType is now .shadowMap.type."),
            this.shadowMap.type = e
        }
    },
    shadowMapCullFace: {
        get: function() {
            return this.shadowMap.cullFace
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderer: .shadowMapCullFace is now .shadowMap.cullFace."),
            this.shadowMap.cullFace = e
        }
    }
}),
Object.defineProperties(n.WebGLRenderTarget.prototype, {
    wrapS: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .wrapS is now .texture.wrapS."),
            this.texture.wrapS
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .wrapS is now .texture.wrapS."),
            this.texture.wrapS = e
        }
    },
    wrapT: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .wrapT is now .texture.wrapT."),
            this.texture.wrapT
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .wrapT is now .texture.wrapT."),
            this.texture.wrapT = e
        }
    },
    magFilter: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .magFilter is now .texture.magFilter."),
            this.texture.magFilter
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .magFilter is now .texture.magFilter."),
            this.texture.magFilter = e
        }
    },
    minFilter: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .minFilter is now .texture.minFilter."),
            this.texture.minFilter
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .minFilter is now .texture.minFilter."),
            this.texture.minFilter = e
        }
    },
    anisotropy: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .anisotropy is now .texture.anisotropy."),
            this.texture.anisotropy
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .anisotropy is now .texture.anisotropy."),
            this.texture.anisotropy = e
        }
    },
    offset: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .offset is now .texture.offset."),
            this.texture.offset
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .offset is now .texture.offset."),
            this.texture.offset = e
        }
    },
    repeat: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .repeat is now .texture.repeat."),
            this.texture.repeat
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .repeat is now .texture.repeat."),
            this.texture.repeat = e
        }
    },
    format: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .format is now .texture.format."),
            this.texture.format
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .format is now .texture.format."),
            this.texture.format = e
        }
    },
    type: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .type is now .texture.type."),
            this.texture.type
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .type is now .texture.type."),
            this.texture.type = e
        }
    },
    generateMipmaps: {
        get: function() {
            return console.warn("THREE.WebGLRenderTarget: .generateMipmaps is now .texture.generateMipmaps."),
            this.texture.generateMipmaps
        },
        set: function(e) {
            console.warn("THREE.WebGLRenderTarget: .generateMipmaps is now .texture.generateMipmaps."),
            this.texture.generateMipmaps = e
        }
    }
}),
n.GeometryUtils = {
    merge: function(e, t, i) {
        console.warn("THREE.GeometryUtils: .merge() has been moved to Geometry. Use geometry.merge( geometry2, matrix, materialIndexOffset ) instead.");
        var r;
        t instanceof n.Mesh && (t.matrixAutoUpdate && t.updateMatrix(),
        r = t.matrix,
        t = t.geometry),
        e.merge(t, r, i)
    },
    center: function(e) {
        return console.warn("THREE.GeometryUtils: .center() has been moved to Geometry. Use geometry.center() instead."),
        e.center()
    }
},
n.ImageUtils = {
    crossOrigin: void 0,
    loadTexture: function(e, t, i, r) {
        console.warn("THREE.ImageUtils.loadTexture has been deprecated. Use THREE.TextureLoader() instead.");
        var o = new n.TextureLoader;
        o.setCrossOrigin(this.crossOrigin);
        var a = o.load(e, i, void 0, r);
        return t && (a.mapping = t),
        a
    },
    loadTextureCube: function(e, t, i, r) {
        console.warn("THREE.ImageUtils.loadTextureCube has been deprecated. Use THREE.CubeTextureLoader() instead.");
        var o = new n.CubeTextureLoader;
        o.setCrossOrigin(this.crossOrigin);
        var a = o.load(e, i, void 0, r);
        return t && (a.mapping = t),
        a
    },
    loadCompressedTexture: function() {
        console.error("THREE.ImageUtils.loadCompressedTexture has been removed. Use THREE.DDSLoader instead.")
    },
    loadCompressedTextureCube: function() {
        console.error("THREE.ImageUtils.loadCompressedTextureCube has been removed. Use THREE.DDSLoader instead.")
    }
},
n.Projector = function() {
    console.error("THREE.Projector has been moved to /examples/js/renderers/Projector.js."),
    this.projectVector = function(e, t) {
        console.warn("THREE.Projector: .projectVector() is now vector.project()."),
        e.project(t)
    }
    ,
    this.unprojectVector = function(e, t) {
        console.warn("THREE.Projector: .unprojectVector() is now vector.unproject()."),
        e.unproject(t)
    }
    ,
    this.pickingRay = function(e, t) {
        console.error("THREE.Projector: .pickingRay() is now raycaster.setFromCamera().")
    }
}
,
n.CanvasRenderer = function() {
    console.error("THREE.CanvasRenderer has been moved to /examples/js/renderers/CanvasRenderer.js"),
    this.domElement = document.createElement("canvas"),
    this.clear = function() {}
    ,
    this.render = function() {}
    ,
    this.setClearColor = function() {}
    ,
    this.setSize = function() {}
}
,
n.MeshFaceMaterial = n.MultiMaterial,
n.CurveUtils = {
    tangentQuadraticBezier: function(e, t, i, n) {
        return 2 * (1 - e) * (i - t) + 2 * e * (n - i)
    },
    tangentCubicBezier: function(e, t, i, n, r) {
        return -3 * t * (1 - e) * (1 - e) + 3 * i * (1 - e) * (1 - e) - 6 * e * i * (1 - e) + 6 * e * n * (1 - e) - 3 * e * e * n + 3 * e * e * r
    },
    tangentSpline: function(e, t, i, n, r) {
        var o = 6 * e * e - 6 * e
          , a = 3 * e * e - 4 * e + 1
          , s = -6 * e * e + 6 * e
          , l = 3 * e * e - 2 * e;
        return o + a + s + l
    },
    interpolate: function(e, t, i, n, r) {
        var o = .5 * (i - e)
          , a = .5 * (n - t)
          , s = r * r
          , l = r * s;
        return (2 * t - 2 * i + o + a) * l + (-3 * t + 3 * i - 2 * o - a) * s + o * r + t
    }
},
n.SceneUtils = {
    createMultiMaterialObject: function(e, t) {
        for (var i = new n.Group, r = 0, o = t.length; r < o; r++)
            i.add(new n.Mesh(e,t[r]));
        return i
    },
    detach: function(e, t, i) {
        e.applyMatrix(t.matrixWorld),
        t.remove(e),
        i.add(e)
    },
    attach: function(e, t, i) {
        var r = new n.Matrix4;
        r.getInverse(i.matrixWorld),
        e.applyMatrix(r),
        t.remove(e),
        i.add(e)
    }
},
n.ShapeUtils = {
    area: function(e) {
        for (var t = e.length, i = 0, n = t - 1, r = 0; r < t; n = r++)
            i += e[n].x * e[r].y - e[r].x * e[n].y;
        return .5 * i
    },
    triangulate: function() {
        function e(e, t, i, n, r, o) {
            var a, s, l, h, c, u, d, p, f;
            if (s = e[o[t]].x,
            l = e[o[t]].y,
            h = e[o[i]].x,
            c = e[o[i]].y,
            u = e[o[n]].x,
            d = e[o[n]].y,
            Number.EPSILON > (h - s) * (d - l) - (c - l) * (u - s))
                return !1;
            var g, m, v, A, y, C, I, b, w, E, x, T, M, S, _;
            for (g = u - h,
            m = d - c,
            v = s - u,
            A = l - d,
            y = h - s,
            C = c - l,
            a = 0; a < r; a++)
                if (p = e[o[a]].x,
                f = e[o[a]].y,
                !(p === s && f === l || p === h && f === c || p === u && f === d) && (I = p - s,
                b = f - l,
                w = p - h,
                E = f - c,
                x = p - u,
                T = f - d,
                _ = g * E - m * w,
                M = y * b - C * I,
                S = v * T - A * x,
                _ >= -Number.EPSILON && S >= -Number.EPSILON && M >= -Number.EPSILON))
                    return !1;
            return !0
        }
        return function(t, i) {
            var r = t.length;
            if (r < 3)
                return null;
            var o, a, s, l = [], h = [], c = [];
            if (n.ShapeUtils.area(t) > 0)
                for (a = 0; a < r; a++)
                    h[a] = a;
            else
                for (a = 0; a < r; a++)
                    h[a] = r - 1 - a;
            var u = r
              , d = 2 * u;
            for (a = u - 1; u > 2; ) {
                if (d-- <= 0)
                    return console.warn("THREE.ShapeUtils: Unable to triangulate polygon! in triangulate()"),
                    i ? c : l;
                if (o = a,
                u <= o && (o = 0),
                a = o + 1,
                u <= a && (a = 0),
                s = a + 1,
                u <= s && (s = 0),
                e(t, o, a, s, u, h)) {
                    var p, f, g, m, v;
                    for (p = h[o],
                    f = h[a],
                    g = h[s],
                    l.push([t[p], t[f], t[g]]),
                    c.push([h[o], h[a], h[s]]),
                    m = a,
                    v = a + 1; v < u; m++,
                    v++)
                        h[m] = h[v];
                    u--,
                    d = 2 * u
                }
            }
            return i ? c : l
        }
    }(),
    triangulateShape: function(e, t) {
        function i(e, t, i) {
            return e.x !== t.x ? e.x < t.x ? e.x <= i.x && i.x <= t.x : t.x <= i.x && i.x <= e.x : e.y < t.y ? e.y <= i.y && i.y <= t.y : t.y <= i.y && i.y <= e.y
        }
        function r(e, t, n, r, o) {
            var a = t.x - e.x
              , s = t.y - e.y
              , l = r.x - n.x
              , h = r.y - n.y
              , c = e.x - n.x
              , u = e.y - n.y
              , d = s * l - a * h
              , p = s * c - a * u;
            if (Math.abs(d) > Number.EPSILON) {
                var f;
                if (d > 0) {
                    if (p < 0 || p > d)
                        return [];
                    if (f = h * c - l * u,
                    f < 0 || f > d)
                        return []
                } else {
                    if (p > 0 || p < d)
                        return [];
                    if (f = h * c - l * u,
                    f > 0 || f < d)
                        return []
                }
                if (0 === f)
                    return !o || 0 !== p && p !== d ? [e] : [];
                if (f === d)
                    return !o || 0 !== p && p !== d ? [t] : [];
                if (0 === p)
                    return [n];
                if (p === d)
                    return [r];
                var g = f / d;
                return [{
                    x: e.x + g * a,
                    y: e.y + g * s
                }]
            }
            if (0 !== p || h * c !== l * u)
                return [];
            var m = 0 === a && 0 === s
              , v = 0 === l && 0 === h;
            if (m && v)
                return e.x !== n.x || e.y !== n.y ? [] : [e];
            if (m)
                return i(n, r, e) ? [e] : [];
            if (v)
                return i(e, t, n) ? [n] : [];
            var A, y, C, I, b, w, E, x;
            return 0 !== a ? (e.x < t.x ? (A = e,
            C = e.x,
            y = t,
            I = t.x) : (A = t,
            C = t.x,
            y = e,
            I = e.x),
            n.x < r.x ? (b = n,
            E = n.x,
            w = r,
            x = r.x) : (b = r,
            E = r.x,
            w = n,
            x = n.x)) : (e.y < t.y ? (A = e,
            C = e.y,
            y = t,
            I = t.y) : (A = t,
            C = t.y,
            y = e,
            I = e.y),
            n.y < r.y ? (b = n,
            E = n.y,
            w = r,
            x = r.y) : (b = r,
            E = r.y,
            w = n,
            x = n.y)),
            C <= E ? I < E ? [] : I === E ? o ? [] : [b] : I <= x ? [b, y] : [b, w] : C > x ? [] : C === x ? o ? [] : [A] : I <= x ? [A, y] : [A, w]
        }
        function o(e, t, i, n) {
            var r = t.x - e.x
              , o = t.y - e.y
              , a = i.x - e.x
              , s = i.y - e.y
              , l = n.x - e.x
              , h = n.y - e.y
              , c = r * s - o * a
              , u = r * h - o * l;
            if (Math.abs(c) > Number.EPSILON) {
                var d = l * s - h * a;
                return c > 0 ? u >= 0 && d >= 0 : u >= 0 || d >= 0
            }
            return u > 0
        }
        function a(e, t) {
            function i(e, t) {
                var i = A.length - 1
                  , n = e - 1;
                n < 0 && (n = i);
                var r = e + 1;
                r > i && (r = 0);
                var a = o(A[e], A[n], A[r], s[t]);
                if (!a)
                    return !1;
                var l = s.length - 1
                  , h = t - 1;
                h < 0 && (h = l);
                var c = t + 1;
                return c > l && (c = 0),
                a = o(s[t], s[h], s[c], A[e]),
                !!a
            }
            function n(e, t) {
                var i, n, o;
                for (i = 0; i < A.length; i++)
                    if (n = i + 1,
                    n %= A.length,
                    o = r(e, t, A[i], A[n], !0),
                    o.length > 0)
                        return !0;
                return !1
            }
            function a(e, i) {
                var n, o, a, s, l;
                for (n = 0; n < y.length; n++)
                    for (o = t[y[n]],
                    a = 0; a < o.length; a++)
                        if (s = a + 1,
                        s %= o.length,
                        l = r(e, i, o[a], o[s], !0),
                        l.length > 0)
                            return !0;
                return !1
            }
            for (var s, l, h, c, u, d, p, f, g, m, v, A = e.concat(), y = [], C = [], I = 0, b = t.length; I < b; I++)
                y.push(I);
            for (var w = 0, E = 2 * y.length; y.length > 0; ) {
                if (E--,
                E < 0) {
                    console.log("Infinite Loop! Holes left:" + y.length + ", Probably Hole outside Shape!");
                    break
                }
                for (h = w; h < A.length; h++) {
                    c = A[h],
                    l = -1;
                    for (var I = 0; I < y.length; I++)
                        if (d = y[I],
                        p = c.x + ":" + c.y + ":" + d,
                        void 0 === C[p]) {
                            s = t[d];
                            for (var x = 0; x < s.length; x++)
                                if (u = s[x],
                                i(h, x) && !n(c, u) && !a(c, u)) {
                                    l = x,
                                    y.splice(I, 1),
                                    f = A.slice(0, h + 1),
                                    g = A.slice(h),
                                    m = s.slice(l),
                                    v = s.slice(0, l + 1),
                                    A = f.concat(m).concat(v).concat(g),
                                    w = h;
                                    break
                                }
                            if (l >= 0)
                                break;
                            C[p] = !0
                        }
                    if (l >= 0)
                        break
                }
            }
            return A
        }
        for (var s, l, h, c, u, d, p = {}, f = e.concat(), g = 0, m = t.length; g < m; g++)
            Array.prototype.push.apply(f, t[g]);
        for (s = 0,
        l = f.length; s < l; s++)
            u = f[s].x + ":" + f[s].y,
            void 0 !== p[u] && console.warn("THREE.Shape: Duplicate point", u),
            p[u] = s;
        var v = a(e, t)
          , A = n.ShapeUtils.triangulate(v, !1);
        for (s = 0,
        l = A.length; s < l; s++)
            for (c = A[s],
            h = 0; h < 3; h++)
                u = c[h].x + ":" + c[h].y,
                d = p[u],
                void 0 !== d && (c[h] = d);
        return A.concat()
    },
    isClockWise: function(e) {
        return n.ShapeUtils.area(e) < 0
    },
    b2: function() {
        function e(e, t) {
            var i = 1 - e;
            return i * i * t
        }
        function t(e, t) {
            return 2 * (1 - e) * e * t
        }
        function i(e, t) {
            return e * e * t
        }
        return function(n, r, o, a) {
            return e(n, r) + t(n, o) + i(n, a)
        }
    }(),
    b3: function() {
        function e(e, t) {
            var i = 1 - e;
            return i * i * i * t
        }
        function t(e, t) {
            var i = 1 - e;
            return 3 * i * i * e * t
        }
        function i(e, t) {
            var i = 1 - e;
            return 3 * i * e * e * t
        }
        function n(e, t) {
            return e * e * e * t
        }
        return function(r, o, a, s, l) {
            return e(r, o) + t(r, a) + i(r, s) + n(r, l)
        }
    }()
},
n.Curve = function() {}
,
n.Curve.prototype = {
    constructor: n.Curve,
    getPoint: function(e) {
        return console.warn("THREE.Curve: Warning, getPoint() not implemented!"),
        null
    },
    getPointAt: function(e) {
        var t = this.getUtoTmapping(e);
        return this.getPoint(t)
    },
    getPoints: function(e) {
        e || (e = 5);
        var t, i = [];
        for (t = 0; t <= e; t++)
            i.push(this.getPoint(t / e));
        return i
    },
    getSpacedPoints: function(e) {
        e || (e = 5);
        var t, i = [];
        for (t = 0; t <= e; t++)
            i.push(this.getPointAt(t / e));
        return i
    },
    getLength: function() {
        var e = this.getLengths();
        return e[e.length - 1]
    },
    getLengths: function(e) {
        if (e || (e = this.__arcLengthDivisions ? this.__arcLengthDivisions : 200),
        this.cacheArcLengths && this.cacheArcLengths.length === e + 1 && !this.needsUpdate)
            return this.cacheArcLengths;
        this.needsUpdate = !1;
        var t, i, n = [], r = this.getPoint(0), o = 0;
        for (n.push(0),
        i = 1; i <= e; i++)
            t = this.getPoint(i / e),
            o += t.distanceTo(r),
            n.push(o),
            r = t;
        return this.cacheArcLengths = n,
        n
    },
    updateArcLengths: function() {
        this.needsUpdate = !0,
        this.getLengths()
    },
    getUtoTmapping: function(e, t) {
        var i, n = this.getLengths(), r = 0, o = n.length;
        i = t ? t : e * n[o - 1];
        for (var a, s = 0, l = o - 1; s <= l; )
            if (r = Math.floor(s + (l - s) / 2),
            a = n[r] - i,
            a < 0)
                s = r + 1;
            else {
                if (!(a > 0)) {
                    l = r;
                    break
                }
                l = r - 1
            }
        if (r = l,
        n[r] === i) {
            var h = r / (o - 1);
            return h
        }
        var c = n[r]
          , u = n[r + 1]
          , d = u - c
          , p = (i - c) / d
          , h = (r + p) / (o - 1);
        return h
    },
    getTangent: function(e) {
        var t = 1e-4
          , i = e - t
          , n = e + t;
        i < 0 && (i = 0),
        n > 1 && (n = 1);
        var r = this.getPoint(i)
          , o = this.getPoint(n)
          , a = o.clone().sub(r);
        return a.normalize()
    },
    getTangentAt: function(e) {
        var t = this.getUtoTmapping(e);
        return this.getTangent(t)
    }
},
n.Curve.create = function(e, t) {
    return e.prototype = Object.create(n.Curve.prototype),
    e.prototype.constructor = e,
    e.prototype.getPoint = t,
    e
}
,
n.CurvePath = function() {
    this.curves = [],
    this.autoClose = !1
}
,
n.CurvePath.prototype = Object.create(n.Curve.prototype),
n.CurvePath.prototype.constructor = n.CurvePath,
n.CurvePath.prototype.add = function(e) {
    this.curves.push(e)
}
,
n.CurvePath.prototype.closePath = function() {
    var e = this.curves[0].getPoint(0)
      , t = this.curves[this.curves.length - 1].getPoint(1);
    e.equals(t) || this.curves.push(new n.LineCurve(t,e))
}
,
n.CurvePath.prototype.getPoint = function(e) {
    for (var t = e * this.getLength(), i = this.getCurveLengths(), n = 0; n < i.length; ) {
        if (i[n] >= t) {
            var r = i[n] - t
              , o = this.curves[n]
              , a = 1 - r / o.getLength();
            return o.getPointAt(a)
        }
        n++
    }
    return null
}
,
n.CurvePath.prototype.getLength = function() {
    var e = this.getCurveLengths();
    return e[e.length - 1]
}
,
n.CurvePath.prototype.getCurveLengths = function() {
    if (this.cacheLengths && this.cacheLengths.length === this.curves.length)
        return this.cacheLengths;
    for (var e = [], t = 0, i = 0, n = this.curves.length; i < n; i++)
        t += this.curves[i].getLength(),
        e.push(t);
    return this.cacheLengths = e,
    e
}
,
n.CurvePath.prototype.createPointsGeometry = function(e) {
    var t = this.getPoints(e);
    return this.createGeometry(t)
}
,
n.CurvePath.prototype.createSpacedPointsGeometry = function(e) {
    var t = this.getSpacedPoints(e);
    return this.createGeometry(t)
}
,
n.CurvePath.prototype.createGeometry = function(e) {
    for (var t = new n.Geometry, i = 0, r = e.length; i < r; i++) {
        var o = e[i];
        t.vertices.push(new n.Vector3(o.x,o.y,o.z || 0))
    }
    return t
}
,
n.Font = function(e) {
    this.data = e
}
,
n.Font.prototype = {
    constructor: n.Font,
    generateShapes: function(e, t, i) {
        function r(e) {
            for (var i = String(e).split(""), n = t / a.resolution, r = 0, s = [], l = 0; l < i.length; l++) {
                var h = o(i[l], n, r);
                r += h.offset,
                s.push(h.path)
            }
            return s
        }
        function o(e, t, r) {
            var o = a.glyphs[e] || a.glyphs["?"];
            if (o) {
                var s, l, h, c, u, d, p, f, g, m, v, A = new n.Path, y = [], C = n.ShapeUtils.b2, I = n.ShapeUtils.b3;
                if (o.o)
                    for (var b = o._cachedOutline || (o._cachedOutline = o.o.split(" ")), w = 0, E = b.length; w < E; ) {
                        var x = b[w++];
                        switch (x) {
                        case "m":
                            s = b[w++] * t + r,
                            l = b[w++] * t,
                            A.moveTo(s, l);
                            break;
                        case "l":
                            s = b[w++] * t + r,
                            l = b[w++] * t,
                            A.lineTo(s, l);
                            break;
                        case "q":
                            if (h = b[w++] * t + r,
                            c = b[w++] * t,
                            p = b[w++] * t + r,
                            f = b[w++] * t,
                            A.quadraticCurveTo(p, f, h, c),
                            v = y[y.length - 1]) {
                                u = v.x,
                                d = v.y;
                                for (var T = 1; T <= i; T++) {
                                    var M = T / i;
                                    C(M, u, p, h),
                                    C(M, d, f, c)
                                }
                            }
                            break;
                        case "b":
                            if (h = b[w++] * t + r,
                            c = b[w++] * t,
                            p = b[w++] * t + r,
                            f = b[w++] * t,
                            g = b[w++] * t + r,
                            m = b[w++] * t,
                            A.bezierCurveTo(p, f, g, m, h, c),
                            v = y[y.length - 1]) {
                                u = v.x,
                                d = v.y;
                                for (var T = 1; T <= i; T++) {
                                    var M = T / i;
                                    I(M, u, p, g, h),
                                    I(M, d, f, m, c)
                                }
                            }
                        }
                    }
                return {
                    offset: o.ha * t,
                    path: A
                }
            }
        }
        void 0 === t && (t = 100),
        void 0 === i && (i = 4);
        for (var a = this.data, s = r(e), l = [], h = 0, c = s.length; h < c; h++)
            Array.prototype.push.apply(l, s[h].toShapes());
        return l
    }
},
n.Path = function(e) {
    n.CurvePath.call(this),
    this.actions = [],
    e && this.fromPoints(e)
}
,
n.Path.prototype = Object.create(n.CurvePath.prototype),
n.Path.prototype.constructor = n.Path,
n.Path.prototype.fromPoints = function(e) {
    this.moveTo(e[0].x, e[0].y);
    for (var t = 1, i = e.length; t < i; t++)
        this.lineTo(e[t].x, e[t].y)
}
,
n.Path.prototype.moveTo = function(e, t) {
    this.actions.push({
        action: "moveTo",
        args: [e, t]
    })
}
,
n.Path.prototype.lineTo = function(e, t) {
    var i = this.actions[this.actions.length - 1].args
      , r = i[i.length - 2]
      , o = i[i.length - 1]
      , a = new n.LineCurve(new n.Vector2(r,o),new n.Vector2(e,t));
    this.curves.push(a),
    this.actions.push({
        action: "lineTo",
        args: [e, t]
    })
}
,
n.Path.prototype.quadraticCurveTo = function(e, t, i, r) {
    var o = this.actions[this.actions.length - 1].args
      , a = o[o.length - 2]
      , s = o[o.length - 1]
      , l = new n.QuadraticBezierCurve(new n.Vector2(a,s),new n.Vector2(e,t),new n.Vector2(i,r));
    this.curves.push(l),
    this.actions.push({
        action: "quadraticCurveTo",
        args: [e, t, i, r]
    })
}
,
n.Path.prototype.bezierCurveTo = function(e, t, i, r, o, a) {
    var s = this.actions[this.actions.length - 1].args
      , l = s[s.length - 2]
      , h = s[s.length - 1]
      , c = new n.CubicBezierCurve(new n.Vector2(l,h),new n.Vector2(e,t),new n.Vector2(i,r),new n.Vector2(o,a));
    this.curves.push(c),
    this.actions.push({
        action: "bezierCurveTo",
        args: [e, t, i, r, o, a]
    })
}
,
n.Path.prototype.splineThru = function(e) {
    var t = Array.prototype.slice.call(arguments)
      , i = this.actions[this.actions.length - 1].args
      , r = i[i.length - 2]
      , o = i[i.length - 1]
      , a = [new n.Vector2(r,o)];
    Array.prototype.push.apply(a, e);
    var s = new n.SplineCurve(a);
    this.curves.push(s),
    this.actions.push({
        action: "splineThru",
        args: t
    })
}
,
n.Path.prototype.arc = function(e, t, i, n, r, o) {
    var a = this.actions[this.actions.length - 1].args
      , s = a[a.length - 2]
      , l = a[a.length - 1];
    this.absarc(e + s, t + l, i, n, r, o)
}
,
n.Path.prototype.absarc = function(e, t, i, n, r, o) {
    this.absellipse(e, t, i, i, n, r, o)
}
,
n.Path.prototype.ellipse = function(e, t, i, n, r, o, a, s) {
    var l = this.actions[this.actions.length - 1].args
      , h = l[l.length - 2]
      , c = l[l.length - 1];
    this.absellipse(e + h, t + c, i, n, r, o, a, s)
}
,
n.Path.prototype.absellipse = function(e, t, i, r, o, a, s, l) {
    var h = [e, t, i, r, o, a, s, l || 0]
      , c = new n.EllipseCurve(e,t,i,r,o,a,s,l);
    this.curves.push(c);
    var u = c.getPoint(1);
    h.push(u.x),
    h.push(u.y),
    this.actions.push({
        action: "ellipse",
        args: h
    })
}
,
n.Path.prototype.getSpacedPoints = function(e) {
    e || (e = 40);
    for (var t = [], i = 0; i < e; i++)
        t.push(this.getPoint(i / e));
    return this.autoClose && t.push(t[0]),
    t
}
,
n.Path.prototype.getPoints = function(e) {
    e = e || 12;
    for (var t, i, r, o, a, s, l, h, c, u, d, p = n.ShapeUtils.b2, f = n.ShapeUtils.b3, g = [], m = 0, v = this.actions.length; m < v; m++) {
        var A = this.actions[m]
          , y = A.action
          , C = A.args;
        switch (y) {
        case "moveTo":
            g.push(new n.Vector2(C[0],C[1]));
            break;
        case "lineTo":
            g.push(new n.Vector2(C[0],C[1]));
            break;
        case "quadraticCurveTo":
            t = C[2],
            i = C[3],
            a = C[0],
            s = C[1],
            g.length > 0 ? (c = g[g.length - 1],
            l = c.x,
            h = c.y) : (c = this.actions[m - 1].args,
            l = c[c.length - 2],
            h = c[c.length - 1]);
            for (var I = 1; I <= e; I++) {
                var b = I / e;
                u = p(b, l, a, t),
                d = p(b, h, s, i),
                g.push(new n.Vector2(u,d))
            }
            break;
        case "bezierCurveTo":
            t = C[4],
            i = C[5],
            a = C[0],
            s = C[1],
            r = C[2],
            o = C[3],
            g.length > 0 ? (c = g[g.length - 1],
            l = c.x,
            h = c.y) : (c = this.actions[m - 1].args,
            l = c[c.length - 2],
            h = c[c.length - 1]);
            for (var I = 1; I <= e; I++) {
                var b = I / e;
                u = f(b, l, a, r, t),
                d = f(b, h, s, o, i),
                g.push(new n.Vector2(u,d))
            }
            break;
        case "splineThru":
            c = this.actions[m - 1].args;
            var w = new n.Vector2(c[c.length - 2],c[c.length - 1])
              , E = [w]
              , x = e * C[0].length;
            E = E.concat(C[0]);
            for (var T = new n.SplineCurve(E), I = 1; I <= x; I++)
                g.push(T.getPointAt(I / x));
            break;
        case "arc":
            for (var M, S = C[0], _ = C[1], P = C[2], R = C[3], L = C[4], O = !!C[5], D = L - R, F = 2 * e, I = 1; I <= F; I++) {
                var b = I / F;
                O || (b = 1 - b),
                M = R + b * D,
                u = S + P * Math.cos(M),
                d = _ + P * Math.sin(M),
                g.push(new n.Vector2(u,d))
            }
            break;
        case "ellipse":
            var M, N, B, S = C[0], _ = C[1], k = C[2], U = C[3], R = C[4], L = C[5], O = !!C[6], V = C[7], D = L - R, F = 2 * e;
            0 !== V && (N = Math.cos(V),
            B = Math.sin(V));
            for (var I = 1; I <= F; I++) {
                var b = I / F;
                if (O || (b = 1 - b),
                M = R + b * D,
                u = S + k * Math.cos(M),
                d = _ + U * Math.sin(M),
                0 !== V) {
                    var z = u
                      , G = d;
                    u = (z - S) * N - (G - _) * B + S,
                    d = (z - S) * B + (G - _) * N + _
                }
                g.push(new n.Vector2(u,d))
            }
        }
    }
    var H = g[g.length - 1];
    return Math.abs(H.x - g[0].x) < Number.EPSILON && Math.abs(H.y - g[0].y) < Number.EPSILON && g.splice(g.length - 1, 1),
    this.autoClose && g.push(g[0]),
    g
}
,
n.Path.prototype.toShapes = function(e, t) {
    function i(e) {
        for (var t = [], i = new n.Path, r = 0, o = e.length; r < o; r++) {
            var a = e[r]
              , s = a.args
              , l = a.action;
            "moveTo" === l && 0 !== i.actions.length && (t.push(i),
            i = new n.Path),
            i[l].apply(i, s)
        }
        return 0 !== i.actions.length && t.push(i),
        t
    }
    function r(e) {
        for (var t = [], i = 0, r = e.length; i < r; i++) {
            var o = e[i]
              , a = new n.Shape;
            a.actions = o.actions,
            a.curves = o.curves,
            t.push(a)
        }
        return t
    }
    function o(e, t) {
        for (var i = t.length, n = !1, r = i - 1, o = 0; o < i; r = o++) {
            var a = t[r]
              , s = t[o]
              , l = s.x - a.x
              , h = s.y - a.y;
            if (Math.abs(h) > Number.EPSILON) {
                if (h < 0 && (a = t[o],
                l = -l,
                s = t[r],
                h = -h),
                e.y < a.y || e.y > s.y)
                    continue;
                if (e.y === a.y) {
                    if (e.x === a.x)
                        return !0
                } else {
                    var c = h * (e.x - a.x) - l * (e.y - a.y);
                    if (0 === c)
                        return !0;
                    if (c < 0)
                        continue;
                    n = !n
                }
            } else {
                if (e.y !== a.y)
                    continue;
                if (s.x <= e.x && e.x <= a.x || a.x <= e.x && e.x <= s.x)
                    return !0
            }
        }
        return n
    }
    var a = n.ShapeUtils.isClockWise
      , s = i(this.actions);
    if (0 === s.length)
        return [];
    if (t === !0)
        return r(s);
    var l, h, c, u = [];
    if (1 === s.length)
        return h = s[0],
        c = new n.Shape,
        c.actions = h.actions,
        c.curves = h.curves,
        u.push(c),
        u;
    var d = !a(s[0].getPoints());
    d = e ? !d : d;
    var p, f = [], g = [], m = [], v = 0;
    g[v] = void 0,
    m[v] = [];
    for (var A = 0, y = s.length; A < y; A++)
        h = s[A],
        p = h.getPoints(),
        l = a(p),
        l = e ? !l : l,
        l ? (!d && g[v] && v++,
        g[v] = {
            s: new n.Shape,
            p: p
        },
        g[v].s.actions = h.actions,
        g[v].s.curves = h.curves,
        d && v++,
        m[v] = []) : m[v].push({
            h: h,
            p: p[0]
        });
    if (!g[0])
        return r(s);
    if (g.length > 1) {
        for (var C = !1, I = [], b = 0, w = g.length; b < w; b++)
            f[b] = [];
        for (var b = 0, w = g.length; b < w; b++)
            for (var E = m[b], x = 0; x < E.length; x++) {
                for (var T = E[x], M = !0, S = 0; S < g.length; S++)
                    o(T.p, g[S].p) && (b !== S && I.push({
                        froms: b,
                        tos: S,
                        hole: x
                    }),
                    M ? (M = !1,
                    f[S].push(T)) : C = !0);
                M && f[b].push(T)
            }
        I.length > 0 && (C || (m = f))
    }
    for (var _, A = 0, P = g.length; A < P; A++) {
        c = g[A].s,
        u.push(c),
        _ = m[A];
        for (var R = 0, L = _.length; R < L; R++)
            c.holes.push(_[R].h)
    }
    return u
}
,
n.Shape = function() {
    n.Path.apply(this, arguments),
    this.holes = []
}
,
n.Shape.prototype = Object.create(n.Path.prototype),
n.Shape.prototype.constructor = n.Shape,
n.Shape.prototype.extrude = function(e) {
    return new n.ExtrudeGeometry(this,e)
}
,
n.Shape.prototype.makeGeometry = function(e) {
    return new n.ShapeGeometry(this,e)
}
,
n.Shape.prototype.getPointsHoles = function(e) {
    for (var t = [], i = 0, n = this.holes.length; i < n; i++)
        t[i] = this.holes[i].getPoints(e);
    return t
}
,
n.Shape.prototype.extractAllPoints = function(e) {
    return {
        shape: this.getPoints(e),
        holes: this.getPointsHoles(e)
    }
}
,
n.Shape.prototype.extractPoints = function(e) {
    return this.extractAllPoints(e)
}
,
n.LineCurve = function(e, t) {
    this.v1 = e,
    this.v2 = t
}
,
n.LineCurve.prototype = Object.create(n.Curve.prototype),
n.LineCurve.prototype.constructor = n.LineCurve,
n.LineCurve.prototype.getPoint = function(e) {
    var t = this.v2.clone().sub(this.v1);
    return t.multiplyScalar(e).add(this.v1),
    t
}
,
n.LineCurve.prototype.getPointAt = function(e) {
    return this.getPoint(e)
}
,
n.LineCurve.prototype.getTangent = function(e) {
    var t = this.v2.clone().sub(this.v1);
    return t.normalize()
}
,
n.QuadraticBezierCurve = function(e, t, i) {
    this.v0 = e,
    this.v1 = t,
    this.v2 = i
}
,
n.QuadraticBezierCurve.prototype = Object.create(n.Curve.prototype),
n.QuadraticBezierCurve.prototype.constructor = n.QuadraticBezierCurve,
n.QuadraticBezierCurve.prototype.getPoint = function(e) {
    var t = n.ShapeUtils.b2;
    return new n.Vector2(t(e, this.v0.x, this.v1.x, this.v2.x),t(e, this.v0.y, this.v1.y, this.v2.y))
}
,
n.QuadraticBezierCurve.prototype.getTangent = function(e) {
    var t = n.CurveUtils.tangentQuadraticBezier;
    return new n.Vector2(t(e, this.v0.x, this.v1.x, this.v2.x),t(e, this.v0.y, this.v1.y, this.v2.y)).normalize()
}
,
n.CubicBezierCurve = function(e, t, i, n) {
    this.v0 = e,
    this.v1 = t,
    this.v2 = i,
    this.v3 = n
}
,
n.CubicBezierCurve.prototype = Object.create(n.Curve.prototype),
n.CubicBezierCurve.prototype.constructor = n.CubicBezierCurve,
n.CubicBezierCurve.prototype.getPoint = function(e) {
    var t = n.ShapeUtils.b3;
    return new n.Vector2(t(e, this.v0.x, this.v1.x, this.v2.x, this.v3.x),t(e, this.v0.y, this.v1.y, this.v2.y, this.v3.y))
}
,
n.CubicBezierCurve.prototype.getTangent = function(e) {
    var t = n.CurveUtils.tangentCubicBezier;
    return new n.Vector2(t(e, this.v0.x, this.v1.x, this.v2.x, this.v3.x),t(e, this.v0.y, this.v1.y, this.v2.y, this.v3.y)).normalize()
}
,
n.SplineCurve = function(e) {
    this.points = void 0 == e ? [] : e
}
,
n.SplineCurve.prototype = Object.create(n.Curve.prototype),
n.SplineCurve.prototype.constructor = n.SplineCurve,
n.SplineCurve.prototype.getPoint = function(e) {
    var t = this.points
      , i = (t.length - 1) * e
      , r = Math.floor(i)
      , o = i - r
      , a = t[0 === r ? r : r - 1]
      , s = t[r]
      , l = t[r > t.length - 2 ? t.length - 1 : r + 1]
      , h = t[r > t.length - 3 ? t.length - 1 : r + 2]
      , c = n.CurveUtils.interpolate;
    return new n.Vector2(c(a.x, s.x, l.x, h.x, o),c(a.y, s.y, l.y, h.y, o))
}
,
n.EllipseCurve = function(e, t, i, n, r, o, a, s) {
    this.aX = e,
    this.aY = t,
    this.xRadius = i,
    this.yRadius = n,
    this.aStartAngle = r,
    this.aEndAngle = o,
    this.aClockwise = a,
    this.aRotation = s || 0
}
,
n.EllipseCurve.prototype = Object.create(n.Curve.prototype),
n.EllipseCurve.prototype.constructor = n.EllipseCurve,
n.EllipseCurve.prototype.getPoint = function(e) {
    var t = this.aEndAngle - this.aStartAngle;
    t < 0 && (t += 2 * Math.PI),
    t > 2 * Math.PI && (t -= 2 * Math.PI);
    var i;
    i = this.aClockwise === !0 ? this.aEndAngle + (1 - e) * (2 * Math.PI - t) : this.aStartAngle + e * t;
    var r = this.aX + this.xRadius * Math.cos(i)
      , o = this.aY + this.yRadius * Math.sin(i);
    if (0 !== this.aRotation) {
        var a = Math.cos(this.aRotation)
          , s = Math.sin(this.aRotation)
          , l = r
          , h = o;
        r = (l - this.aX) * a - (h - this.aY) * s + this.aX,
        o = (l - this.aX) * s + (h - this.aY) * a + this.aY
    }
    return new n.Vector2(r,o)
}
,
n.ArcCurve = function(e, t, i, r, o, a) {
    n.EllipseCurve.call(this, e, t, i, i, r, o, a)
}
,
n.ArcCurve.prototype = Object.create(n.EllipseCurve.prototype),
n.ArcCurve.prototype.constructor = n.ArcCurve,
n.LineCurve3 = n.Curve.create(function(e, t) {
    this.v1 = e,
    this.v2 = t
}, function(e) {
    var t = new n.Vector3;
    return t.subVectors(this.v2, this.v1),
    t.multiplyScalar(e),
    t.add(this.v1),
    t
}),
n.QuadraticBezierCurve3 = n.Curve.create(function(e, t, i) {
    this.v0 = e,
    this.v1 = t,
    this.v2 = i
}, function(e) {
    var t = n.ShapeUtils.b2;
    return new n.Vector3(t(e, this.v0.x, this.v1.x, this.v2.x),t(e, this.v0.y, this.v1.y, this.v2.y),t(e, this.v0.z, this.v1.z, this.v2.z))
}),
n.CubicBezierCurve3 = n.Curve.create(function(e, t, i, n) {
    this.v0 = e,
    this.v1 = t,
    this.v2 = i,
    this.v3 = n
}, function(e) {
    var t = n.ShapeUtils.b3;
    return new n.Vector3(t(e, this.v0.x, this.v1.x, this.v2.x, this.v3.x),t(e, this.v0.y, this.v1.y, this.v2.y, this.v3.y),t(e, this.v0.z, this.v1.z, this.v2.z, this.v3.z))
}),
n.SplineCurve3 = n.Curve.create(function(e) {
    console.warn("THREE.SplineCurve3 will be deprecated. Please use THREE.CatmullRomCurve3"),
    this.points = void 0 == e ? [] : e
}, function(e) {
    var t = this.points
      , i = (t.length - 1) * e
      , r = Math.floor(i)
      , o = i - r
      , a = t[0 == r ? r : r - 1]
      , s = t[r]
      , l = t[r > t.length - 2 ? t.length - 1 : r + 1]
      , h = t[r > t.length - 3 ? t.length - 1 : r + 2]
      , c = n.CurveUtils.interpolate;
    return new n.Vector3(c(a.x, s.x, l.x, h.x, o),c(a.y, s.y, l.y, h.y, o),c(a.z, s.z, l.z, h.z, o))
}),
n.CatmullRomCurve3 = function() {
    function e() {}
    var t = new n.Vector3
      , i = new e
      , r = new e
      , o = new e;
    return e.prototype.init = function(e, t, i, n) {
        this.c0 = e,
        this.c1 = i,
        this.c2 = -3 * e + 3 * t - 2 * i - n,
        this.c3 = 2 * e - 2 * t + i + n
    }
    ,
    e.prototype.initNonuniformCatmullRom = function(e, t, i, n, r, o, a) {
        var s = (t - e) / r - (i - e) / (r + o) + (i - t) / o
          , l = (i - t) / o - (n - t) / (o + a) + (n - i) / a;
        s *= o,
        l *= o,
        this.init(t, i, s, l)
    }
    ,
    e.prototype.initCatmullRom = function(e, t, i, n, r) {
        this.init(t, i, r * (i - e), r * (n - t))
    }
    ,
    e.prototype.calc = function(e) {
        var t = e * e
          , i = t * e;
        return this.c0 + this.c1 * e + this.c2 * t + this.c3 * i
    }
    ,
    n.Curve.create(function(e) {
        this.points = e || [],
        this.closed = !1
    }, function(e) {
        var a, s, l, h, c = this.points;
        h = c.length,
        h < 2 && console.log("duh, you need at least 2 points"),
        a = (h - (this.closed ? 0 : 1)) * e,
        s = Math.floor(a),
        l = a - s,
        this.closed ? s += s > 0 ? 0 : (Math.floor(Math.abs(s) / c.length) + 1) * c.length : 0 === l && s === h - 1 && (s = h - 2,
        l = 1);
        var u, d, p, f;
        if (this.closed || s > 0 ? u = c[(s - 1) % h] : (t.subVectors(c[0], c[1]).add(c[0]),
        u = t),
        d = c[s % h],
        p = c[(s + 1) % h],
        this.closed || s + 2 < h ? f = c[(s + 2) % h] : (t.subVectors(c[h - 1], c[h - 2]).add(c[h - 1]),
        f = t),
        void 0 === this.type || "centripetal" === this.type || "chordal" === this.type) {
            var g = "chordal" === this.type ? .5 : .25
              , m = Math.pow(u.distanceToSquared(d), g)
              , v = Math.pow(d.distanceToSquared(p), g)
              , A = Math.pow(p.distanceToSquared(f), g);
            v < 1e-4 && (v = 1),
            m < 1e-4 && (m = v),
            A < 1e-4 && (A = v),
            i.initNonuniformCatmullRom(u.x, d.x, p.x, f.x, m, v, A),
            r.initNonuniformCatmullRom(u.y, d.y, p.y, f.y, m, v, A),
            o.initNonuniformCatmullRom(u.z, d.z, p.z, f.z, m, v, A)
        } else if ("catmullrom" === this.type) {
            var y = void 0 !== this.tension ? this.tension : .5;
            i.initCatmullRom(u.x, d.x, p.x, f.x, y),
            r.initCatmullRom(u.y, d.y, p.y, f.y, y),
            o.initCatmullRom(u.z, d.z, p.z, f.z, y)
        }
        var C = new n.Vector3(i.calc(l),r.calc(l),o.calc(l));
        return C
    })
}(),
n.ClosedSplineCurve3 = function(e) {
    console.warn("THREE.ClosedSplineCurve3 has been deprecated. Please use THREE.CatmullRomCurve3."),
    n.CatmullRomCurve3.call(this, e),
    this.type = "catmullrom",
    this.closed = !0
}
,
n.ClosedSplineCurve3.prototype = Object.create(n.CatmullRomCurve3.prototype),
n.BoxGeometry = function(e, t, i, r, o, a) {
    n.Geometry.call(this),
    this.type = "BoxGeometry",
    this.parameters = {
        width: e,
        height: t,
        depth: i,
        widthSegments: r,
        heightSegments: o,
        depthSegments: a
    },
    this.fromBufferGeometry(new n.BoxBufferGeometry(e,t,i,r,o,a)),
    this.mergeVertices()
}
,
n.BoxGeometry.prototype = Object.create(n.Geometry.prototype),
n.BoxGeometry.prototype.constructor = n.BoxGeometry,
n.CubeGeometry = n.BoxGeometry,
n.BoxBufferGeometry = function(e, t, i, r, o, a) {
    function s(e, t, i) {
        var n = 0;
        return n += e * t * 2,
        n += e * i * 2,
        n += i * t * 2,
        4 * n
    }
    function l(e, t, i, r, o, a, s, l, c, u, I) {
        for (var b = a / c, w = s / u, E = a / 2, x = s / 2, T = l / 2, M = c + 1, S = u + 1, _ = 0, P = 0, R = new n.Vector3, L = 0; L < S; L++)
            for (var O = L * w - x, D = 0; D < M; D++) {
                var F = D * b - E;
                R[e] = F * r,
                R[t] = O * o,
                R[i] = T,
                p[m] = R.x,
                p[m + 1] = R.y,
                p[m + 2] = R.z,
                R[e] = 0,
                R[t] = 0,
                R[i] = l > 0 ? 1 : -1,
                f[m] = R.x,
                f[m + 1] = R.y,
                f[m + 2] = R.z,
                g[v] = D / c,
                g[v + 1] = 1 - L / u,
                m += 3,
                v += 2,
                _ += 1
            }
        for (L = 0; L < u; L++)
            for (D = 0; D < c; D++) {
                var N = y + D + M * L
                  , B = y + D + M * (L + 1)
                  , k = y + (D + 1) + M * (L + 1)
                  , U = y + (D + 1) + M * L;
                d[A] = N,
                d[A + 1] = B,
                d[A + 2] = U,
                d[A + 3] = B,
                d[A + 4] = k,
                d[A + 5] = U,
                A += 6,
                P += 6
            }
        h.addGroup(C, P, I),
        C += P,
        y += _
    }
    n.BufferGeometry.call(this),
    this.type = "BoxBufferGeometry",
    this.parameters = {
        width: e,
        height: t,
        depth: i,
        widthSegments: r,
        heightSegments: o,
        depthSegments: a
    };
    var h = this;
    r = Math.floor(r) || 1,
    o = Math.floor(o) || 1,
    a = Math.floor(a) || 1;
    var c = s(r, o, a)
      , u = c / 4 * 6
      , d = new (u > 65535 ? Uint32Array : Uint16Array)(u)
      , p = new Float32Array(3 * c)
      , f = new Float32Array(3 * c)
      , g = new Float32Array(2 * c)
      , m = 0
      , v = 0
      , A = 0
      , y = 0
      , C = 0;
    l("z", "y", "x", -1, -1, i, t, e, a, o, 0),
    l("z", "y", "x", 1, -1, i, t, -e, a, o, 1),
    l("x", "z", "y", 1, 1, e, i, t, r, a, 2),
    l("x", "z", "y", 1, -1, e, i, -t, r, a, 3),
    l("x", "y", "z", 1, -1, e, t, i, r, o, 4),
    l("x", "y", "z", -1, -1, e, t, -i, r, o, 5),
    this.setIndex(new n.BufferAttribute(d,1)),
    this.addAttribute("position", new n.BufferAttribute(p,3)),
    this.addAttribute("normal", new n.BufferAttribute(f,3)),
    this.addAttribute("uv", new n.BufferAttribute(g,2))
}
,
n.BoxBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.BoxBufferGeometry.prototype.constructor = n.BoxBufferGeometry,
n.CircleGeometry = function(e, t, i, r) {
    n.Geometry.call(this),
    this.type = "CircleGeometry",
    this.parameters = {
        radius: e,
        segments: t,
        thetaStart: i,
        thetaLength: r
    },
    this.fromBufferGeometry(new n.CircleBufferGeometry(e,t,i,r))
}
,
n.CircleGeometry.prototype = Object.create(n.Geometry.prototype),
n.CircleGeometry.prototype.constructor = n.CircleGeometry,
n.CircleBufferGeometry = function(e, t, i, r) {
    n.BufferGeometry.call(this),
    this.type = "CircleBufferGeometry",
    this.parameters = {
        radius: e,
        segments: t,
        thetaStart: i,
        thetaLength: r
    },
    e = e || 50,
    t = void 0 !== t ? Math.max(3, t) : 8,
    i = void 0 !== i ? i : 0,
    r = void 0 !== r ? r : 2 * Math.PI;
    var o = t + 2
      , a = new Float32Array(3 * o)
      , s = new Float32Array(3 * o)
      , l = new Float32Array(2 * o);
    s[2] = 1,
    l[0] = .5,
    l[1] = .5;
    for (var h = 0, c = 3, u = 2; h <= t; h++,
    c += 3,
    u += 2) {
        var d = i + h / t * r;
        a[c] = e * Math.cos(d),
        a[c + 1] = e * Math.sin(d),
        s[c + 2] = 1,
        l[u] = (a[c] / e + 1) / 2,
        l[u + 1] = (a[c + 1] / e + 1) / 2
    }
    for (var p = [], c = 1; c <= t; c++)
        p.push(c, c + 1, 0);
    this.setIndex(new n.BufferAttribute(new Uint16Array(p),1)),
    this.addAttribute("position", new n.BufferAttribute(a,3)),
    this.addAttribute("normal", new n.BufferAttribute(s,3)),
    this.addAttribute("uv", new n.BufferAttribute(l,2)),
    this.boundingSphere = new n.Sphere(new n.Vector3,e)
}
,
n.CircleBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.CircleBufferGeometry.prototype.constructor = n.CircleBufferGeometry,
n.CylinderBufferGeometry = function(e, t, i, r, o, a, s, l) {
    function h() {
        var e = (r + 1) * (o + 1);
        return a === !1 && (e += 2 * (r + 1) + 2 * r),
        e
    }
    function c() {
        var e = r * o * 2 * 3;
        return a === !1 && (e += 2 * r * 3),
        e
    }
    function u() {
        var a, h, c = new n.Vector3, u = new n.Vector3, d = (t - e) / i;
        for (h = 0; h <= o; h++) {
            var p = []
              , f = h / o
              , w = f * (t - e) + e;
            for (a = 0; a <= r; a++) {
                var E = a / r;
                u.x = w * Math.sin(E * l + s),
                u.y = -f * i + b,
                u.z = w * Math.cos(E * l + s),
                m.setXYZ(y, u.x, u.y, u.z),
                c.copy(u),
                (0 === e && 0 === h || 0 === t && h === o) && (c.x = Math.sin(E * l + s),
                c.z = Math.cos(E * l + s)),
                c.setY(Math.sqrt(c.x * c.x + c.z * c.z) * d).normalize(),
                v.setXYZ(y, c.x, c.y, c.z),
                A.setXY(y, E, 1 - f),
                p.push(y),
                y++
            }
            I.push(p)
        }
        for (a = 0; a < r; a++)
            for (h = 0; h < o; h++) {
                var x = I[h][a]
                  , T = I[h + 1][a]
                  , M = I[h + 1][a + 1]
                  , S = I[h][a + 1];
                g.setX(C, x),
                C++,
                g.setX(C, T),
                C++,
                g.setX(C, S),
                C++,
                g.setX(C, T),
                C++,
                g.setX(C, M),
                C++,
                g.setX(C, S),
                C++
            }
    }
    function d(i) {
        var o, a, h, c = new n.Vector2, u = new n.Vector3, d = i === !0 ? e : t, p = i === !0 ? 1 : -1;
        for (a = y,
        o = 1; o <= r; o++)
            m.setXYZ(y, 0, b * p, 0),
            v.setXYZ(y, 0, p, 0),
            i === !0 ? (c.x = o / r,
            c.y = 0) : (c.x = (o - 1) / r,
            c.y = 1),
            A.setXY(y, c.x, c.y),
            y++;
        for (h = y,
        o = 0; o <= r; o++) {
            var f = o / r;
            u.x = d * Math.sin(f * l + s),
            u.y = b * p,
            u.z = d * Math.cos(f * l + s),
            m.setXYZ(y, u.x, u.y, u.z),
            v.setXYZ(y, 0, p, 0),
            A.setXY(y, f, i === !0 ? 1 : 0),
            y++
        }
        for (o = 0; o < r; o++) {
            var I = a + o
              , w = h + o;
            i === !0 ? (g.setX(C, w),
            C++,
            g.setX(C, w + 1),
            C++,
            g.setX(C, I),
            C++) : (g.setX(C, w + 1),
            C++,
            g.setX(C, w),
            C++,
            g.setX(C, I),
            C++)
        }
    }
    n.BufferGeometry.call(this),
    this.type = "CylinderBufferGeometry",
    this.parameters = {
        radiusTop: e,
        radiusBottom: t,
        height: i,
        radialSegments: r,
        heightSegments: o,
        openEnded: a,
        thetaStart: s,
        thetaLength: l
    },
    e = void 0 !== e ? e : 20,
    t = void 0 !== t ? t : 20,
    i = void 0 !== i ? i : 100,
    r = Math.floor(r) || 8,
    o = Math.floor(o) || 1,
    a = void 0 !== a && a,
    s = void 0 !== s ? s : 0,
    l = void 0 !== l ? l : 2 * Math.PI;
    var p = h()
      , f = c()
      , g = new n.BufferAttribute(new (f > 65535 ? Uint32Array : Uint16Array)(f),1)
      , m = new n.BufferAttribute(new Float32Array(3 * p),3)
      , v = new n.BufferAttribute(new Float32Array(3 * p),3)
      , A = new n.BufferAttribute(new Float32Array(2 * p),2)
      , y = 0
      , C = 0
      , I = []
      , b = i / 2;
    u(),
    a === !1 && (e > 0 && d(!0),
    t > 0 && d(!1)),
    this.setIndex(g),
    this.addAttribute("position", m),
    this.addAttribute("normal", v),
    this.addAttribute("uv", A)
}
,
n.CylinderBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.CylinderBufferGeometry.prototype.constructor = n.CylinderBufferGeometry,
n.CylinderGeometry = function(e, t, i, r, o, a, s, l) {
    n.Geometry.call(this),
    this.type = "CylinderGeometry",
    this.parameters = {
        radiusTop: e,
        radiusBottom: t,
        height: i,
        radialSegments: r,
        heightSegments: o,
        openEnded: a,
        thetaStart: s,
        thetaLength: l
    },
    this.fromBufferGeometry(new n.CylinderBufferGeometry(e,t,i,r,o,a,s,l)),
    this.mergeVertices()
}
,
n.CylinderGeometry.prototype = Object.create(n.Geometry.prototype),
n.CylinderGeometry.prototype.constructor = n.CylinderGeometry,
n.EdgesGeometry = function(e, t) {
    function i(e, t) {
        return e - t
    }
    n.BufferGeometry.call(this),
    t = void 0 !== t ? t : 1;
    var r, o = Math.cos(n.Math.degToRad(t)), a = [0, 0], s = {}, l = ["a", "b", "c"];
    e instanceof n.BufferGeometry ? (r = new n.Geometry,
    r.fromBufferGeometry(e)) : r = e.clone(),
    r.mergeVertices(),
    r.computeFaceNormals();
    for (var h = r.vertices, c = r.faces, u = 0, d = c.length; u < d; u++)
        for (var p = c[u], f = 0; f < 3; f++) {
            a[0] = p[l[f]],
            a[1] = p[l[(f + 1) % 3]],
            a.sort(i);
            var g = a.toString();
            void 0 === s[g] ? s[g] = {
                vert1: a[0],
                vert2: a[1],
                face1: u,
                face2: void 0
            } : s[g].face2 = u
        }
    var m = [];
    for (var g in s) {
        var v = s[g];
        if (void 0 === v.face2 || c[v.face1].normal.dot(c[v.face2].normal) <= o) {
            var A = h[v.vert1];
            m.push(A.x),
            m.push(A.y),
            m.push(A.z),
            A = h[v.vert2],
            m.push(A.x),
            m.push(A.y),
            m.push(A.z)
        }
    }
    this.addAttribute("position", new n.BufferAttribute(new Float32Array(m),3))
}
,
n.EdgesGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.EdgesGeometry.prototype.constructor = n.EdgesGeometry,
n.ExtrudeGeometry = function(e, t) {
    return "undefined" == typeof e ? void (e = []) : (n.Geometry.call(this),
    this.type = "ExtrudeGeometry",
    e = Array.isArray(e) ? e : [e],
    this.addShapeList(e, t),
    void this.computeFaceNormals())
}
;
n.ExtrudeGeometry.prototype = Object.create(n.Geometry.prototype);
n.ExtrudeGeometry.prototype.constructor = n.ExtrudeGeometry,
n.ExtrudeGeometry.prototype.addShapeList = function(e, t) {
    for (var i = e.length, n = 0; n < i; n++) {
        var r = e[n];
        this.addShape(r, t)
    }
}
,
n.ExtrudeGeometry.prototype.addShape = function(e, t) {
    function i(e, t, i) {
        return t || console.error("THREE.ExtrudeGeometry: vec does not exist"),
        t.clone().multiplyScalar(i).add(e)
    }
    function r(e, t, i) {
        var r, o, a = 1, s = e.x - t.x, l = e.y - t.y, h = i.x - e.x, c = i.y - e.y, u = s * s + l * l, d = s * c - l * h;
        if (Math.abs(d) > Number.EPSILON) {
            var p = Math.sqrt(u)
              , f = Math.sqrt(h * h + c * c)
              , g = t.x - l / p
              , m = t.y + s / p
              , v = i.x - c / f
              , A = i.y + h / f
              , y = ((v - g) * c - (A - m) * h) / (s * c - l * h);
            r = g + s * y - e.x,
            o = m + l * y - e.y;
            var C = r * r + o * o;
            if (C <= 2)
                return new n.Vector2(r,o);
            a = Math.sqrt(C / 2)
        } else {
            var I = !1;
            s > Number.EPSILON ? h > Number.EPSILON && (I = !0) : s < -Number.EPSILON ? h < -Number.EPSILON && (I = !0) : Math.sign(l) === Math.sign(c) && (I = !0),
            I ? (r = -l,
            o = s,
            a = Math.sqrt(u)) : (r = s,
            o = l,
            a = Math.sqrt(u / 2))
        }
        return new n.Vector2(r / a,o / a)
    }
    function o() {
        if (C) {
            var e = 0
              , t = H * e;
            for (Y = 0; Y < W; Y++)
                G = F[Y],
                h(G[2] + t, G[1] + t, G[0] + t);
            for (e = b + 2 * y,
            t = H * e,
            Y = 0; Y < W; Y++)
                G = F[Y],
                h(G[0] + t, G[1] + t, G[2] + t)
        } else {
            for (Y = 0; Y < W; Y++)
                G = F[Y],
                h(G[2], G[1], G[0]);
            for (Y = 0; Y < W; Y++)
                G = F[Y],
                h(G[0] + H * b, G[1] + H * b, G[2] + H * b)
        }
    }
    function a() {
        var e = 0;
        for (s(N, e),
        e += N.length,
        M = 0,
        S = O.length; M < S; M++)
            T = O[M],
            s(T, e),
            e += T.length
    }
    function s(e, t) {
        var i, n;
        for (Y = e.length; --Y >= 0; ) {
            i = Y,
            n = Y - 1,
            n < 0 && (n = e.length - 1);
            var r = 0
              , o = b + 2 * y;
            for (r = 0; r < o; r++) {
                var a = H * r
                  , s = H * (r + 1)
                  , l = t + i + a
                  , h = t + n + a
                  , u = t + n + s
                  , d = t + i + s;
                c(l, h, u, d, e, r, o, i, n)
            }
        }
    }
    function l(e, t, i) {
        _.vertices.push(new n.Vector3(e,t,i))
    }
    function h(e, t, i) {
        e += P,
        t += P,
        i += P,
        _.faces.push(new n.Face3(e,t,i,null,null,0));
        var r = x.generateTopUV(_, e, t, i);
        _.faceVertexUvs[0].push(r)
    }
    function c(e, t, i, r, o, a, s, l, h) {
        e += P,
        t += P,
        i += P,
        r += P,
        _.faces.push(new n.Face3(e,t,r,null,null,1)),
        _.faces.push(new n.Face3(t,i,r,null,null,1));
        var c = x.generateSideWallUV(_, e, t, i, r);
        _.faceVertexUvs[0].push([c[0], c[1], c[3]]),
        _.faceVertexUvs[0].push([c[1], c[2], c[3]])
    }
    var u, d, p, f, g, m = void 0 !== t.amount ? t.amount : 100, v = void 0 !== t.bevelThickness ? t.bevelThickness : 6, A = void 0 !== t.bevelSize ? t.bevelSize : v - 2, y = void 0 !== t.bevelSegments ? t.bevelSegments : 3, C = void 0 === t.bevelEnabled || t.bevelEnabled, I = void 0 !== t.curveSegments ? t.curveSegments : 12, b = void 0 !== t.steps ? t.steps : 1, w = t.extrudePath, E = !1, x = void 0 !== t.UVGenerator ? t.UVGenerator : n.ExtrudeGeometry.WorldUVGenerator;
    w && (u = w.getSpacedPoints(b),
    E = !0,
    C = !1,
    d = void 0 !== t.frames ? t.frames : new n.TubeGeometry.FrenetFrames(w,b,!1),
    p = new n.Vector3,
    f = new n.Vector3,
    g = new n.Vector3),
    C || (y = 0,
    v = 0,
    A = 0);
    var T, M, S, _ = this, P = this.vertices.length, R = e.extractPoints(I), L = R.shape, O = R.holes, D = !n.ShapeUtils.isClockWise(L);
    if (D) {
        for (L = L.reverse(),
        M = 0,
        S = O.length; M < S; M++)
            T = O[M],
            n.ShapeUtils.isClockWise(T) && (O[M] = T.reverse());
        D = !1
    }
    var F = n.ShapeUtils.triangulateShape(L, O)
      , N = L;
    for (M = 0,
    S = O.length; M < S; M++)
        T = O[M],
        L = L.concat(T);
    for (var B, k, U, V, z, G, H = L.length, W = F.length, j = [], Y = 0, X = N.length, Z = X - 1, Q = Y + 1; Y < X; Y++,
    Z++,
    Q++)
        Z === X && (Z = 0),
        Q === X && (Q = 0),
        j[Y] = r(N[Y], N[Z], N[Q]);
    var q, K = [], J = j.concat();
    for (M = 0,
    S = O.length; M < S; M++) {
        for (T = O[M],
        q = [],
        Y = 0,
        X = T.length,
        Z = X - 1,
        Q = Y + 1; Y < X; Y++,
        Z++,
        Q++)
            Z === X && (Z = 0),
            Q === X && (Q = 0),
            q[Y] = r(T[Y], T[Z], T[Q]);
        K.push(q),
        J = J.concat(q)
    }
    for (B = 0; B < y; B++) {
        for (U = B / y,
        V = v * (1 - U),
        k = A * Math.sin(U * Math.PI / 2),
        Y = 0,
        X = N.length; Y < X; Y++)
            z = i(N[Y], j[Y], k),
            l(z.x, z.y, -V);
        for (M = 0,
        S = O.length; M < S; M++)
            for (T = O[M],
            q = K[M],
            Y = 0,
            X = T.length; Y < X; Y++)
                z = i(T[Y], q[Y], k),
                l(z.x, z.y, -V)
    }
    for (k = A,
    Y = 0; Y < H; Y++)
        z = C ? i(L[Y], J[Y], k) : L[Y],
        E ? (f.copy(d.normals[0]).multiplyScalar(z.x),
        p.copy(d.binormals[0]).multiplyScalar(z.y),
        g.copy(u[0]).add(f).add(p),
        l(g.x, g.y, g.z)) : l(z.x, z.y, 0);
    var $;
    for ($ = 1; $ <= b; $++)
        for (Y = 0; Y < H; Y++)
            z = C ? i(L[Y], J[Y], k) : L[Y],
            E ? (f.copy(d.normals[$]).multiplyScalar(z.x),
            p.copy(d.binormals[$]).multiplyScalar(z.y),
            g.copy(u[$]).add(f).add(p),
            l(g.x, g.y, g.z)) : l(z.x, z.y, m / b * $);
    for (B = y - 1; B >= 0; B--) {
        for (U = B / y,
        V = v * (1 - U),
        k = A * Math.sin(U * Math.PI / 2),
        Y = 0,
        X = N.length; Y < X; Y++)
            z = i(N[Y], j[Y], k),
            l(z.x, z.y, m + V);
        for (M = 0,
        S = O.length; M < S; M++)
            for (T = O[M],
            q = K[M],
            Y = 0,
            X = T.length; Y < X; Y++)
                z = i(T[Y], q[Y], k),
                E ? l(z.x, z.y + u[b - 1].y, u[b - 1].x + V) : l(z.x, z.y, m + V)
    }
    o(),
    a()
}
,
n.ExtrudeGeometry.WorldUVGenerator = {
    generateTopUV: function(e, t, i, r) {
        var o = e.vertices
          , a = o[t]
          , s = o[i]
          , l = o[r];
        return [new n.Vector2(a.x,a.y), new n.Vector2(s.x,s.y), new n.Vector2(l.x,l.y)]
    },
    generateSideWallUV: function(e, t, i, r, o) {
        var a = e.vertices
          , s = a[t]
          , l = a[i]
          , h = a[r]
          , c = a[o];
        return Math.abs(s.y - l.y) < .01 ? [new n.Vector2(s.x,1 - s.z), new n.Vector2(l.x,1 - l.z), new n.Vector2(h.x,1 - h.z), new n.Vector2(c.x,1 - c.z)] : [new n.Vector2(s.y,1 - s.z), new n.Vector2(l.y,1 - l.z), new n.Vector2(h.y,1 - h.z), new n.Vector2(c.y,1 - c.z)]
    }
},
n.ShapeGeometry = function(e, t) {
    n.Geometry.call(this),
    this.type = "ShapeGeometry",
    Array.isArray(e) === !1 && (e = [e]),
    this.addShapeList(e, t),
    this.computeFaceNormals()
}
,
n.ShapeGeometry.prototype = Object.create(n.Geometry.prototype),
n.ShapeGeometry.prototype.constructor = n.ShapeGeometry,
n.ShapeGeometry.prototype.addShapeList = function(e, t) {
    for (var i = 0, n = e.length; i < n; i++)
        this.addShape(e[i], t);
    return this
}
,
n.ShapeGeometry.prototype.addShape = function(e, t) {
    void 0 === t && (t = {});
    var i, r, o, a = void 0 !== t.curveSegments ? t.curveSegments : 12, s = t.material, l = void 0 === t.UVGenerator ? n.ExtrudeGeometry.WorldUVGenerator : t.UVGenerator, h = this.vertices.length, c = e.extractPoints(a), u = c.shape, d = c.holes, p = !n.ShapeUtils.isClockWise(u);
    if (p) {
        for (u = u.reverse(),
        i = 0,
        r = d.length; i < r; i++)
            o = d[i],
            n.ShapeUtils.isClockWise(o) && (d[i] = o.reverse());
        p = !1
    }
    var f = n.ShapeUtils.triangulateShape(u, d);
    for (i = 0,
    r = d.length; i < r; i++)
        o = d[i],
        u = u.concat(o);
    var g, m, v = u.length, A = f.length;
    for (i = 0; i < v; i++)
        g = u[i],
        this.vertices.push(new n.Vector3(g.x,g.y,0));
    for (i = 0; i < A; i++) {
        m = f[i];
        var y = m[0] + h
          , C = m[1] + h
          , I = m[2] + h;
        this.faces.push(new n.Face3(y,C,I,null,null,s)),
        this.faceVertexUvs[0].push(l.generateTopUV(this, y, C, I))
    }
}
,
n.LatheBufferGeometry = function(e, t, i, r) {
    n.BufferGeometry.call(this),
    this.type = "LatheBufferGeometry",
    this.parameters = {
        points: e,
        segments: t,
        phiStart: i,
        phiLength: r
    },
    t = Math.floor(t) || 12,
    i = i || 0,
    r = r || 2 * Math.PI,
    r = n.Math.clamp(r, 0, 2 * Math.PI);
    var o, a, s, l = (t + 1) * e.length, h = t * e.length * 2 * 3, c = new n.BufferAttribute(new (h > 65535 ? Uint32Array : Uint16Array)(h),1), u = new n.BufferAttribute(new Float32Array(3 * l),3), d = new n.BufferAttribute(new Float32Array(2 * l),2), p = 0, f = 0, g = (1 / (e.length - 1),
    1 / t), m = new n.Vector3, v = new n.Vector2;
    for (a = 0; a <= t; a++) {
        var A = i + a * g * r
          , y = Math.sin(A)
          , C = Math.cos(A);
        for (s = 0; s <= e.length - 1; s++)
            m.x = e[s].x * y,
            m.y = e[s].y,
            m.z = e[s].x * C,
            u.setXYZ(p, m.x, m.y, m.z),
            v.x = a / t,
            v.y = s / (e.length - 1),
            d.setXY(p, v.x, v.y),
            p++
    }
    for (a = 0; a < t; a++)
        for (s = 0; s < e.length - 1; s++) {
            o = s + a * e.length;
            var I = o
              , b = o + e.length
              , w = o + e.length + 1
              , E = o + 1;
            c.setX(f, I),
            f++,
            c.setX(f, b),
            f++,
            c.setX(f, E),
            f++,
            c.setX(f, b),
            f++,
            c.setX(f, w),
            f++,
            c.setX(f, E),
            f++
        }
    if (this.setIndex(c),
    this.addAttribute("position", u),
    this.addAttribute("uv", d),
    this.computeVertexNormals(),
    r === 2 * Math.PI) {
        var x = this.attributes.normal.array
          , T = new n.Vector3
          , M = new n.Vector3
          , S = new n.Vector3;
        for (o = t * e.length * 3,
        a = 0,
        s = 0; a < e.length; a++,
        s += 3)
            T.x = x[s + 0],
            T.y = x[s + 1],
            T.z = x[s + 2],
            M.x = x[o + s + 0],
            M.y = x[o + s + 1],
            M.z = x[o + s + 2],
            S.addVectors(T, M).normalize(),
            x[s + 0] = x[o + s + 0] = S.x,
            x[s + 1] = x[o + s + 1] = S.y,
            x[s + 2] = x[o + s + 2] = S.z
    }
}
,
n.LatheBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.LatheBufferGeometry.prototype.constructor = n.LatheBufferGeometry,
n.LatheGeometry = function(e, t, i, r) {
    n.Geometry.call(this),
    this.type = "LatheGeometry",
    this.parameters = {
        points: e,
        segments: t,
        phiStart: i,
        phiLength: r
    },
    this.fromBufferGeometry(new n.LatheBufferGeometry(e,t,i,r)),
    this.mergeVertices()
}
,
n.LatheGeometry.prototype = Object.create(n.Geometry.prototype),
n.LatheGeometry.prototype.constructor = n.LatheGeometry,
n.PlaneGeometry = function(e, t, i, r) {
    n.Geometry.call(this),
    this.type = "PlaneGeometry",
    this.parameters = {
        width: e,
        height: t,
        widthSegments: i,
        heightSegments: r
    },
    this.fromBufferGeometry(new n.PlaneBufferGeometry(e,t,i,r))
}
,
n.PlaneGeometry.prototype = Object.create(n.Geometry.prototype),
n.PlaneGeometry.prototype.constructor = n.PlaneGeometry,
n.PlaneBufferGeometry = function(e, t, i, r) {
    n.BufferGeometry.call(this),
    this.type = "PlaneBufferGeometry",
    this.parameters = {
        width: e,
        height: t,
        widthSegments: i,
        heightSegments: r
    };
    for (var o = e / 2, a = t / 2, s = Math.floor(i) || 1, l = Math.floor(r) || 1, h = s + 1, c = l + 1, u = e / s, d = t / l, p = new Float32Array(h * c * 3), f = new Float32Array(h * c * 3), g = new Float32Array(h * c * 2), m = 0, v = 0, A = 0; A < c; A++)
        for (var y = A * d - a, C = 0; C < h; C++) {
            var I = C * u - o;
            p[m] = I,
            p[m + 1] = -y,
            f[m + 2] = 1,
            g[v] = C / s,
            g[v + 1] = 1 - A / l,
            m += 3,
            v += 2
        }
    m = 0;
    for (var b = new (p.length / 3 > 65535 ? Uint32Array : Uint16Array)(s * l * 6), A = 0; A < l; A++)
        for (var C = 0; C < s; C++) {
            var w = C + h * A
              , E = C + h * (A + 1)
              , x = C + 1 + h * (A + 1)
              , T = C + 1 + h * A;
            b[m] = w,
            b[m + 1] = E,
            b[m + 2] = T,
            b[m + 3] = E,
            b[m + 4] = x,
            b[m + 5] = T,
            m += 6
        }
    this.setIndex(new n.BufferAttribute(b,1)),
    this.addAttribute("position", new n.BufferAttribute(p,3)),
    this.addAttribute("normal", new n.BufferAttribute(f,3)),
    this.addAttribute("uv", new n.BufferAttribute(g,2))
}
,
n.PlaneBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.PlaneBufferGeometry.prototype.constructor = n.PlaneBufferGeometry,
n.RingBufferGeometry = function(e, t, i, r, o, a) {
    n.BufferGeometry.call(this),
    this.type = "RingBufferGeometry",
    this.parameters = {
        innerRadius: e,
        outerRadius: t,
        thetaSegments: i,
        phiSegments: r,
        thetaStart: o,
        thetaLength: a
    },
    e = e || 20,
    t = t || 50,
    o = void 0 !== o ? o : 0,
    a = void 0 !== a ? a : 2 * Math.PI,
    i = void 0 !== i ? Math.max(3, i) : 8,
    r = void 0 !== r ? Math.max(1, r) : 1;
    var s, l, h, c = (i + 1) * (r + 1), u = i * r * 2 * 3, d = new n.BufferAttribute(new (u > 65535 ? Uint32Array : Uint16Array)(u),1), p = new n.BufferAttribute(new Float32Array(3 * c),3), f = new n.BufferAttribute(new Float32Array(3 * c),3), g = new n.BufferAttribute(new Float32Array(2 * c),2), m = 0, v = 0, A = e, y = (t - e) / r, C = new n.Vector3, I = new n.Vector2;
    for (l = 0; l <= r; l++) {
        for (h = 0; h <= i; h++)
            s = o + h / i * a,
            C.x = A * Math.cos(s),
            C.y = A * Math.sin(s),
            p.setXYZ(m, C.x, C.y, C.z),
            f.setXYZ(m, 0, 0, 1),
            I.x = (C.x / t + 1) / 2,
            I.y = (C.y / t + 1) / 2,
            g.setXY(m, I.x, I.y),
            m++;
        A += y
    }
    for (l = 0; l < r; l++) {
        var b = l * (i + 1);
        for (h = 0; h < i; h++) {
            s = h + b;
            var w = s
              , E = s + i + 1
              , x = s + i + 2
              , T = s + 1;
            d.setX(v, w),
            v++,
            d.setX(v, E),
            v++,
            d.setX(v, x),
            v++,
            d.setX(v, w),
            v++,
            d.setX(v, x),
            v++,
            d.setX(v, T),
            v++
        }
    }
    this.setIndex(d),
    this.addAttribute("position", p),
    this.addAttribute("normal", f),
    this.addAttribute("uv", g)
}
,
n.RingBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.RingBufferGeometry.prototype.constructor = n.RingBufferGeometry,
n.RingGeometry = function(e, t, i, r, o, a) {
    n.Geometry.call(this),
    this.type = "RingGeometry",
    this.parameters = {
        innerRadius: e,
        outerRadius: t,
        thetaSegments: i,
        phiSegments: r,
        thetaStart: o,
        thetaLength: a
    },
    this.fromBufferGeometry(new n.RingBufferGeometry(e,t,i,r,o,a))
}
,
n.RingGeometry.prototype = Object.create(n.Geometry.prototype),
n.RingGeometry.prototype.constructor = n.RingGeometry,
n.SphereGeometry = function(e, t, i, r, o, a, s) {
    n.Geometry.call(this),
    this.type = "SphereGeometry",
    this.parameters = {
        radius: e,
        widthSegments: t,
        heightSegments: i,
        phiStart: r,
        phiLength: o,
        thetaStart: a,
        thetaLength: s
    },
    this.fromBufferGeometry(new n.SphereBufferGeometry(e,t,i,r,o,a,s))
}
,
n.SphereGeometry.prototype = Object.create(n.Geometry.prototype),
n.SphereGeometry.prototype.constructor = n.SphereGeometry,
n.SphereBufferGeometry = function(e, t, i, r, o, a, s) {
    n.BufferGeometry.call(this),
    this.type = "SphereBufferGeometry",
    this.parameters = {
        radius: e,
        widthSegments: t,
        heightSegments: i,
        phiStart: r,
        phiLength: o,
        thetaStart: a,
        thetaLength: s
    },
    e = e || 50,
    t = Math.max(3, Math.floor(t) || 8),
    i = Math.max(2, Math.floor(i) || 6),
    r = void 0 !== r ? r : 0,
    o = void 0 !== o ? o : 2 * Math.PI,
    a = void 0 !== a ? a : 0,
    s = void 0 !== s ? s : Math.PI;
    for (var l = a + s, h = (t + 1) * (i + 1), c = new n.BufferAttribute(new Float32Array(3 * h),3), u = new n.BufferAttribute(new Float32Array(3 * h),3), d = new n.BufferAttribute(new Float32Array(2 * h),2), p = 0, f = [], g = new n.Vector3, m = 0; m <= i; m++) {
        for (var v = [], A = m / i, y = 0; y <= t; y++) {
            var C = y / t
              , I = -e * Math.cos(r + C * o) * Math.sin(a + A * s)
              , b = e * Math.cos(a + A * s)
              , w = e * Math.sin(r + C * o) * Math.sin(a + A * s);
            g.set(I, b, w).normalize(),
            c.setXYZ(p, I, b, w),
            u.setXYZ(p, g.x, g.y, g.z),
            d.setXY(p, C, 1 - A),
            v.push(p),
            p++
        }
        f.push(v)
    }
    for (var E = [], m = 0; m < i; m++)
        for (var y = 0; y < t; y++) {
            var x = f[m][y + 1]
              , T = f[m][y]
              , M = f[m + 1][y]
              , S = f[m + 1][y + 1];
            (0 !== m || a > 0) && E.push(x, T, S),
            (m !== i - 1 || l < Math.PI) && E.push(T, M, S)
        }
    this.setIndex(new (c.count > 65535 ? n.Uint32Attribute : n.Uint16Attribute)(E,1)),
    this.addAttribute("position", c),
    this.addAttribute("normal", u),
    this.addAttribute("uv", d),
    this.boundingSphere = new n.Sphere(new n.Vector3,e)
}
,
n.SphereBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.SphereBufferGeometry.prototype.constructor = n.SphereBufferGeometry,
n.TextGeometry = function(e, t) {
    t = t || {};
    var i = t.font;
    if (i instanceof n.Font == !1)
        return console.error("THREE.TextGeometry: font parameter is not an instance of THREE.Font."),
        new n.Geometry;
    var r = i.generateShapes(e, t.size, t.curveSegments);
    t.amount = void 0 !== t.height ? t.height : 50,
    void 0 === t.bevelThickness && (t.bevelThickness = 10),
    void 0 === t.bevelSize && (t.bevelSize = 8),
    void 0 === t.bevelEnabled && (t.bevelEnabled = !1),
    n.ExtrudeGeometry.call(this, r, t),
    this.type = "TextGeometry"
}
,
n.TextGeometry.prototype = Object.create(n.ExtrudeGeometry.prototype),
n.TextGeometry.prototype.constructor = n.TextGeometry,
n.TorusBufferGeometry = function(e, t, i, r, o) {
    n.BufferGeometry.call(this),
    this.type = "TorusBufferGeometry",
    this.parameters = {
        radius: e,
        tube: t,
        radialSegments: i,
        tubularSegments: r,
        arc: o
    },
    e = e || 100,
    t = t || 40,
    i = Math.floor(i) || 8,
    r = Math.floor(r) || 6,
    o = o || 2 * Math.PI;
    var a, s, l = (i + 1) * (r + 1), h = i * r * 2 * 3, c = new (h > 65535 ? Uint32Array : Uint16Array)(h), u = new Float32Array(3 * l), d = new Float32Array(3 * l), p = new Float32Array(2 * l), f = 0, g = 0, m = 0, v = new n.Vector3, A = new n.Vector3, y = new n.Vector3;
    for (a = 0; a <= i; a++)
        for (s = 0; s <= r; s++) {
            var C = s / r * o
              , I = a / i * Math.PI * 2;
            A.x = (e + t * Math.cos(I)) * Math.cos(C),
            A.y = (e + t * Math.cos(I)) * Math.sin(C),
            A.z = t * Math.sin(I),
            u[f] = A.x,
            u[f + 1] = A.y,
            u[f + 2] = A.z,
            v.x = e * Math.cos(C),
            v.y = e * Math.sin(C),
            y.subVectors(A, v).normalize(),
            d[f] = y.x,
            d[f + 1] = y.y,
            d[f + 2] = y.z,
            p[g] = s / r,
            p[g + 1] = a / i,
            f += 3,
            g += 2
        }
    for (a = 1; a <= i; a++)
        for (s = 1; s <= r; s++) {
            var b = (r + 1) * a + s - 1
              , w = (r + 1) * (a - 1) + s - 1
              , E = (r + 1) * (a - 1) + s
              , x = (r + 1) * a + s;
            c[m] = b,
            c[m + 1] = w,
            c[m + 2] = x,
            c[m + 3] = w,
            c[m + 4] = E,
            c[m + 5] = x,
            m += 6
        }
    this.setIndex(new n.BufferAttribute(c,1)),
    this.addAttribute("position", new n.BufferAttribute(u,3)),
    this.addAttribute("normal", new n.BufferAttribute(d,3)),
    this.addAttribute("uv", new n.BufferAttribute(p,2))
}
,
n.TorusBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.TorusBufferGeometry.prototype.constructor = n.TorusBufferGeometry,
n.TorusGeometry = function(e, t, i, r, o) {
    n.Geometry.call(this),
    this.type = "TorusGeometry",
    this.parameters = {
        radius: e,
        tube: t,
        radialSegments: i,
        tubularSegments: r,
        arc: o
    },
    this.fromBufferGeometry(new n.TorusBufferGeometry(e,t,i,r,o))
}
,
n.TorusGeometry.prototype = Object.create(n.Geometry.prototype),
n.TorusGeometry.prototype.constructor = n.TorusGeometry,
n.TorusKnotBufferGeometry = function(e, t, i, r, o, a) {
    function s(e, t, i, n, r) {
        var o = Math.cos(e)
          , a = Math.sin(e)
          , s = i / t * e
          , l = Math.cos(s);
        r.x = n * (2 + l) * .5 * o,
        r.y = n * (2 + l) * a * .5,
        r.z = n * Math.sin(s) * .5
    }
    n.BufferGeometry.call(this),
    this.type = "TorusKnotBufferGeometry",
    this.parameters = {
        radius: e,
        tube: t,
        tubularSegments: i,
        radialSegments: r,
        p: o,
        q: a
    },
    e = e || 100,
    t = t || 40,
    i = Math.floor(i) || 64,
    r = Math.floor(r) || 8,
    o = o || 2,
    a = a || 3;
    var l, h, c = (r + 1) * (i + 1), u = r * i * 2 * 3, d = new n.BufferAttribute(new (u > 65535 ? Uint32Array : Uint16Array)(u),1), p = new n.BufferAttribute(new Float32Array(3 * c),3), f = new n.BufferAttribute(new Float32Array(3 * c),3), g = new n.BufferAttribute(new Float32Array(2 * c),2), m = 0, v = 0, A = new n.Vector3, y = new n.Vector3, C = new n.Vector2, I = new n.Vector3, b = new n.Vector3, w = new n.Vector3, E = new n.Vector3, x = new n.Vector3;
    for (l = 0; l <= i; ++l) {
        var T = l / i * o * Math.PI * 2;
        for (s(T, o, a, e, I),
        s(T + .01, o, a, e, b),
        E.subVectors(b, I),
        x.addVectors(b, I),
        w.crossVectors(E, x),
        x.crossVectors(w, E),
        w.normalize(),
        x.normalize(),
        h = 0; h <= r; ++h) {
            var M = h / r * Math.PI * 2
              , S = -t * Math.cos(M)
              , _ = t * Math.sin(M);
            A.x = I.x + (S * x.x + _ * w.x),
            A.y = I.y + (S * x.y + _ * w.y),
            A.z = I.z + (S * x.z + _ * w.z),
            p.setXYZ(m, A.x, A.y, A.z),
            y.subVectors(A, I).normalize(),
            f.setXYZ(m, y.x, y.y, y.z),
            C.x = l / i,
            C.y = h / r,
            g.setXY(m, C.x, C.y),
            m++
        }
    }
    for (h = 1; h <= i; h++)
        for (l = 1; l <= r; l++) {
            var P = (r + 1) * (h - 1) + (l - 1)
              , R = (r + 1) * h + (l - 1)
              , L = (r + 1) * h + l
              , O = (r + 1) * (h - 1) + l;
            d.setX(v, P),
            v++,
            d.setX(v, R),
            v++,
            d.setX(v, O),
            v++,
            d.setX(v, R),
            v++,
            d.setX(v, L),
            v++,
            d.setX(v, O),
            v++
        }
    this.setIndex(d),
    this.addAttribute("position", p),
    this.addAttribute("normal", f),
    this.addAttribute("uv", g)
}
,
n.TorusKnotBufferGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.TorusKnotBufferGeometry.prototype.constructor = n.TorusKnotBufferGeometry,
n.TorusKnotGeometry = function(e, t, i, r, o, a, s) {
    n.Geometry.call(this),
    this.type = "TorusKnotGeometry",
    this.parameters = {
        radius: e,
        tube: t,
        tubularSegments: i,
        radialSegments: r,
        p: o,
        q: a
    },
    void 0 !== s && console.warn("THREE.TorusKnotGeometry: heightScale has been deprecated. Use .scale( x, y, z ) instead."),
    this.fromBufferGeometry(new n.TorusKnotBufferGeometry(e,t,i,r,o,a)),
    this.mergeVertices()
}
,
n.TorusKnotGeometry.prototype = Object.create(n.Geometry.prototype),
n.TorusKnotGeometry.prototype.constructor = n.TorusKnotGeometry,
n.TubeGeometry = function(e, t, i, r, o, a) {
    function s(e, t, i) {
        return P.vertices.push(new n.Vector3(e,t,i)) - 1
    }
    n.Geometry.call(this),
    this.type = "TubeGeometry",
    this.parameters = {
        path: e,
        segments: t,
        radius: i,
        radialSegments: r,
        closed: o,
        taper: a
    },
    t = t || 64,
    i = i || 1,
    r = r || 8,
    o = o || !1,
    a = a || n.TubeGeometry.NoTaper;
    var l, h, c, u, d, p, f, g, m, v, A, y, C, I, b, w, E, x, T, M, S, _ = [], P = this, R = t + 1, L = new n.Vector3, O = new n.TubeGeometry.FrenetFrames(e,t,o), D = O.tangents, F = O.normals, N = O.binormals;
    for (this.tangents = D,
    this.normals = F,
    this.binormals = N,
    v = 0; v < R; v++)
        for (_[v] = [],
        u = v / (R - 1),
        m = e.getPointAt(u),
        l = D[v],
        h = F[v],
        c = N[v],
        p = i * a(u),
        A = 0; A < r; A++)
            d = A / r * 2 * Math.PI,
            f = -p * Math.cos(d),
            g = p * Math.sin(d),
            L.copy(m),
            L.x += f * h.x + g * c.x,
            L.y += f * h.y + g * c.y,
            L.z += f * h.z + g * c.z,
            _[v][A] = s(L.x, L.y, L.z);
    for (v = 0; v < t; v++)
        for (A = 0; A < r; A++)
            y = o ? (v + 1) % t : v + 1,
            C = (A + 1) % r,
            I = _[v][A],
            b = _[y][A],
            w = _[y][C],
            E = _[v][C],
            x = new n.Vector2(v / t,A / r),
            T = new n.Vector2((v + 1) / t,A / r),
            M = new n.Vector2((v + 1) / t,(A + 1) / r),
            S = new n.Vector2(v / t,(A + 1) / r),
            this.faces.push(new n.Face3(I,b,E)),
            this.faceVertexUvs[0].push([x, T, S]),
            this.faces.push(new n.Face3(b,w,E)),
            this.faceVertexUvs[0].push([T.clone(), M, S.clone()]);
    this.computeFaceNormals(),
    this.computeVertexNormals()
}
,
n.TubeGeometry.prototype = Object.create(n.Geometry.prototype),
n.TubeGeometry.prototype.constructor = n.TubeGeometry,
n.TubeGeometry.NoTaper = function(e) {
    return 1
}
,
n.TubeGeometry.SinusoidalTaper = function(e) {
    return Math.sin(Math.PI * e)
}
,
n.TubeGeometry.FrenetFrames = function(e, t, i) {
    function r() {
        f[0] = new n.Vector3,
        g[0] = new n.Vector3,
        a = Number.MAX_VALUE,
        s = Math.abs(p[0].x),
        l = Math.abs(p[0].y),
        h = Math.abs(p[0].z),
        s <= a && (a = s,
        d.set(1, 0, 0)),
        l <= a && (a = l,
        d.set(0, 1, 0)),
        h <= a && d.set(0, 0, 1),
        m.crossVectors(p[0], d).normalize(),
        f[0].crossVectors(p[0], m),
        g[0].crossVectors(p[0], f[0])
    }
    var o, a, s, l, h, c, u, d = new n.Vector3, p = [], f = [], g = [], m = new n.Vector3, v = new n.Matrix4, A = t + 1;
    for (this.tangents = p,
    this.normals = f,
    this.binormals = g,
    c = 0; c < A; c++)
        u = c / (A - 1),
        p[c] = e.getTangentAt(u),
        p[c].normalize();
    for (r(),
    c = 1; c < A; c++)
        f[c] = f[c - 1].clone(),
        g[c] = g[c - 1].clone(),
        m.crossVectors(p[c - 1], p[c]),
        m.length() > Number.EPSILON && (m.normalize(),
        o = Math.acos(n.Math.clamp(p[c - 1].dot(p[c]), -1, 1)),
        f[c].applyMatrix4(v.makeRotationAxis(m, o))),
        g[c].crossVectors(p[c], f[c]);
    if (i)
        for (o = Math.acos(n.Math.clamp(f[0].dot(f[A - 1]), -1, 1)),
        o /= A - 1,
        p[0].dot(m.crossVectors(f[0], f[A - 1])) > 0 && (o = -o),
        c = 1; c < A; c++)
            f[c].applyMatrix4(v.makeRotationAxis(p[c], o * c)),
            g[c].crossVectors(p[c], f[c])
}
,
n.PolyhedronGeometry = function(e, t, i, r) {
    function o(e) {
        var t = e.normalize().clone();
        t.index = u.vertices.push(t) - 1;
        var i = l(e) / 2 / Math.PI + .5
          , r = h(e) / Math.PI + .5;
        return t.uv = new n.Vector2(i,1 - r),
        t
    }
    function a(e, t, i, r) {
        var o = new n.Face3(e.index,t.index,i.index,[e.clone(), t.clone(), i.clone()],void 0,r);
        u.faces.push(o),
        C.copy(e).add(t).add(i).divideScalar(3);
        var a = l(C);
        u.faceVertexUvs[0].push([c(e.uv, e, a), c(t.uv, t, a), c(i.uv, i, a)])
    }
    function s(e, t) {
        for (var i = Math.pow(2, t), n = o(u.vertices[e.a]), r = o(u.vertices[e.b]), s = o(u.vertices[e.c]), l = [], h = e.materialIndex, c = 0; c <= i; c++) {
            l[c] = [];
            for (var d = o(n.clone().lerp(s, c / i)), p = o(r.clone().lerp(s, c / i)), f = i - c, g = 0; g <= f; g++)
                0 === g && c === i ? l[c][g] = d : l[c][g] = o(d.clone().lerp(p, g / f))
        }
        for (var c = 0; c < i; c++)
            for (var g = 0; g < 2 * (i - c) - 1; g++) {
                var m = Math.floor(g / 2);
                g % 2 === 0 ? a(l[c][m + 1], l[c + 1][m], l[c][m], h) : a(l[c][m + 1], l[c + 1][m + 1], l[c + 1][m], h)
            }
    }
    function l(e) {
        return Math.atan2(e.z, -e.x)
    }
    function h(e) {
        return Math.atan2(-e.y, Math.sqrt(e.x * e.x + e.z * e.z))
    }
    function c(e, t, i) {
        return i < 0 && 1 === e.x && (e = new n.Vector2(e.x - 1,e.y)),
        0 === t.x && 0 === t.z && (e = new n.Vector2(i / 2 / Math.PI + .5,e.y)),
        e.clone()
    }
    n.Geometry.call(this),
    this.type = "PolyhedronGeometry",
    this.parameters = {
        vertices: e,
        indices: t,
        radius: i,
        detail: r
    },
    i = i || 1,
    r = r || 0;
    for (var u = this, d = 0, p = e.length; d < p; d += 3)
        o(new n.Vector3(e[d],e[d + 1],e[d + 2]));
    for (var f = this.vertices, g = [], d = 0, m = 0, p = t.length; d < p; d += 3,
    m++) {
        var v = f[t[d]]
          , A = f[t[d + 1]]
          , y = f[t[d + 2]];
        g[m] = new n.Face3(v.index,A.index,y.index,[v.clone(), A.clone(), y.clone()],void 0,m)
    }
    for (var C = new n.Vector3, d = 0, p = g.length; d < p; d++)
        s(g[d], r);
    for (var d = 0, p = this.faceVertexUvs[0].length; d < p; d++) {
        var I = this.faceVertexUvs[0][d]
          , b = I[0].x
          , w = I[1].x
          , E = I[2].x
          , x = Math.max(b, w, E)
          , T = Math.min(b, w, E);
        x > .9 && T < .1 && (b < .2 && (I[0].x += 1),
        w < .2 && (I[1].x += 1),
        E < .2 && (I[2].x += 1))
    }
    for (var d = 0, p = this.vertices.length; d < p; d++)
        this.vertices[d].multiplyScalar(i);
    this.mergeVertices(),
    this.computeFaceNormals(),
    this.boundingSphere = new n.Sphere(new n.Vector3,i)
}
,
n.PolyhedronGeometry.prototype = Object.create(n.Geometry.prototype),
n.PolyhedronGeometry.prototype.constructor = n.PolyhedronGeometry,
n.DodecahedronGeometry = function(e, t) {
    var i = (1 + Math.sqrt(5)) / 2
      , r = 1 / i
      , o = [-1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 0, -r, -i, 0, -r, i, 0, r, -i, 0, r, i, -r, -i, 0, -r, i, 0, r, -i, 0, r, i, 0, -i, 0, -r, i, 0, -r, -i, 0, r, i, 0, r]
      , a = [3, 11, 7, 3, 7, 15, 3, 15, 13, 7, 19, 17, 7, 17, 6, 7, 6, 15, 17, 4, 8, 17, 8, 10, 17, 10, 6, 8, 0, 16, 8, 16, 2, 8, 2, 10, 0, 12, 1, 0, 1, 18, 0, 18, 16, 6, 10, 2, 6, 2, 13, 6, 13, 15, 2, 16, 18, 2, 18, 3, 2, 3, 13, 18, 1, 9, 18, 9, 11, 18, 11, 3, 4, 14, 12, 4, 12, 0, 4, 0, 8, 11, 9, 5, 11, 5, 19, 11, 19, 7, 19, 5, 14, 19, 14, 4, 19, 4, 17, 1, 12, 14, 1, 14, 5, 1, 5, 9];
    n.PolyhedronGeometry.call(this, o, a, e, t),
    this.type = "DodecahedronGeometry",
    this.parameters = {
        radius: e,
        detail: t
    }
}
,
n.DodecahedronGeometry.prototype = Object.create(n.PolyhedronGeometry.prototype),
n.DodecahedronGeometry.prototype.constructor = n.DodecahedronGeometry,
n.IcosahedronGeometry = function(e, t) {
    var i = (1 + Math.sqrt(5)) / 2
      , r = [-1, i, 0, 1, i, 0, -1, -i, 0, 1, -i, 0, 0, -1, i, 0, 1, i, 0, -1, -i, 0, 1, -i, i, 0, -1, i, 0, 1, -i, 0, -1, -i, 0, 1]
      , o = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
    n.PolyhedronGeometry.call(this, r, o, e, t),
    this.type = "IcosahedronGeometry",
    this.parameters = {
        radius: e,
        detail: t
    }
}
,
n.IcosahedronGeometry.prototype = Object.create(n.PolyhedronGeometry.prototype),
n.IcosahedronGeometry.prototype.constructor = n.IcosahedronGeometry,
n.OctahedronGeometry = function(e, t) {
    var i = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]
      , r = [0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2];
    n.PolyhedronGeometry.call(this, i, r, e, t),
    this.type = "OctahedronGeometry",
    this.parameters = {
        radius: e,
        detail: t
    }
}
,
n.OctahedronGeometry.prototype = Object.create(n.PolyhedronGeometry.prototype),
n.OctahedronGeometry.prototype.constructor = n.OctahedronGeometry,
n.TetrahedronGeometry = function(e, t) {
    var i = [1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1]
      , r = [2, 1, 0, 0, 3, 2, 1, 3, 0, 2, 3, 1];
    n.PolyhedronGeometry.call(this, i, r, e, t),
    this.type = "TetrahedronGeometry",
    this.parameters = {
        radius: e,
        detail: t
    }
}
,
n.TetrahedronGeometry.prototype = Object.create(n.PolyhedronGeometry.prototype),
n.TetrahedronGeometry.prototype.constructor = n.TetrahedronGeometry,
n.ParametricGeometry = function(e, t, i) {
    n.Geometry.call(this),
    this.type = "ParametricGeometry",
    this.parameters = {
        func: e,
        slices: t,
        stacks: i
    };
    var r, o, a, s, l, h = this.vertices, c = this.faces, u = this.faceVertexUvs[0], d = t + 1;
    for (r = 0; r <= i; r++)
        for (l = r / i,
        o = 0; o <= t; o++)
            s = o / t,
            a = e(s, l),
            h.push(a);
    var p, f, g, m, v, A, y, C;
    for (r = 0; r < i; r++)
        for (o = 0; o < t; o++)
            p = r * d + o,
            f = r * d + o + 1,
            g = (r + 1) * d + o + 1,
            m = (r + 1) * d + o,
            v = new n.Vector2(o / t,r / i),
            A = new n.Vector2((o + 1) / t,r / i),
            y = new n.Vector2((o + 1) / t,(r + 1) / i),
            C = new n.Vector2(o / t,(r + 1) / i),
            c.push(new n.Face3(p,f,m)),
            u.push([v, A, C]),
            c.push(new n.Face3(f,g,m)),
            u.push([A.clone(), y, C.clone()]);
    this.computeFaceNormals(),
    this.computeVertexNormals()
}
,
n.ParametricGeometry.prototype = Object.create(n.Geometry.prototype),
n.ParametricGeometry.prototype.constructor = n.ParametricGeometry,
n.WireframeGeometry = function(e) {
    function t(e, t) {
        return e - t
    }
    n.BufferGeometry.call(this);
    var i = [0, 0]
      , r = {}
      , o = ["a", "b", "c"];
    if (e instanceof n.Geometry) {
        for (var a = e.vertices, s = e.faces, l = 0, h = new Uint32Array(6 * s.length), c = 0, u = s.length; c < u; c++)
            for (var d = s[c], p = 0; p < 3; p++) {
                i[0] = d[o[p]],
                i[1] = d[o[(p + 1) % 3]],
                i.sort(t);
                var f = i.toString();
                void 0 === r[f] && (h[2 * l] = i[0],
                h[2 * l + 1] = i[1],
                r[f] = !0,
                l++)
            }
        for (var g = new Float32Array(2 * l * 3), c = 0, u = l; c < u; c++)
            for (var p = 0; p < 2; p++) {
                var m = a[h[2 * c + p]]
                  , v = 6 * c + 3 * p;
                g[v + 0] = m.x,
                g[v + 1] = m.y,
                g[v + 2] = m.z
            }
        this.addAttribute("position", new n.BufferAttribute(g,3))
    } else if (e instanceof n.BufferGeometry)
        if (null !== e.index) {
            var A = e.index.array
              , a = e.attributes.position
              , y = e.groups
              , l = 0;
            0 === y.length && e.addGroup(0, A.length);
            for (var h = new Uint32Array(2 * A.length), C = 0, I = y.length; C < I; ++C)
                for (var b = y[C], w = b.start, E = b.count, c = w, x = w + E; c < x; c += 3)
                    for (var p = 0; p < 3; p++) {
                        i[0] = A[c + p],
                        i[1] = A[c + (p + 1) % 3],
                        i.sort(t);
                        var f = i.toString();
                        void 0 === r[f] && (h[2 * l] = i[0],
                        h[2 * l + 1] = i[1],
                        r[f] = !0,
                        l++)
                    }
            for (var g = new Float32Array(2 * l * 3), c = 0, u = l; c < u; c++)
                for (var p = 0; p < 2; p++) {
                    var v = 6 * c + 3 * p
                      , T = h[2 * c + p];
                    g[v + 0] = a.getX(T),
                    g[v + 1] = a.getY(T),
                    g[v + 2] = a.getZ(T)
                }
            this.addAttribute("position", new n.BufferAttribute(g,3))
        } else {
            for (var a = e.attributes.position.array, l = a.length / 3, M = l / 3, g = new Float32Array(2 * l * 3), c = 0, u = M; c < u; c++)
                for (var p = 0; p < 3; p++) {
                    var v = 18 * c + 6 * p
                      , S = 9 * c + 3 * p;
                    g[v + 0] = a[S],
                    g[v + 1] = a[S + 1],
                    g[v + 2] = a[S + 2];
                    var T = 9 * c + 3 * ((p + 1) % 3);
                    g[v + 3] = a[T],
                    g[v + 4] = a[T + 1],
                    g[v + 5] = a[T + 2]
                }
            this.addAttribute("position", new n.BufferAttribute(g,3))
        }
}
,
n.WireframeGeometry.prototype = Object.create(n.BufferGeometry.prototype),
n.WireframeGeometry.prototype.constructor = n.WireframeGeometry,
n.AxisHelper = function(e) {
    e = e || 1;
    var t = new Float32Array([0, 0, 0, e, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, e])
      , i = new Float32Array([1, 0, 0, 1, .6, 0, 0, 1, 0, .6, 1, 0, 0, 0, 1, 0, .6, 1])
      , r = new n.BufferGeometry;
    r.addAttribute("position", new n.BufferAttribute(t,3)),
    r.addAttribute("color", new n.BufferAttribute(i,3));
    var o = new n.LineBasicMaterial({
        vertexColors: n.VertexColors
    });
    n.LineSegments.call(this, r, o)
}
,
n.AxisHelper.prototype = Object.create(n.LineSegments.prototype),
n.AxisHelper.prototype.constructor = n.AxisHelper,
n.ArrowHelper = function() {
    var e = new n.Geometry;
    e.vertices.push(new n.Vector3(0,0,0), new n.Vector3(0,1,0));
    var t = new n.CylinderGeometry(0,.5,1,5,1);
    return t.translate(0, -.5, 0),
    function(i, r, o, a, s, l) {
        n.Object3D.call(this),
        void 0 === a && (a = 16776960),
        void 0 === o && (o = 1),
        void 0 === s && (s = .2 * o),
        void 0 === l && (l = .2 * s),
        this.position.copy(r),
        this.line = new n.Line(e,new n.LineBasicMaterial({
            color: a
        })),
        this.line.matrixAutoUpdate = !1,
        this.add(this.line),
        this.cone = new n.Mesh(t,new n.MeshBasicMaterial({
            color: a
        })),
        this.cone.matrixAutoUpdate = !1,
        this.add(this.cone),
        this.setDirection(i),
        this.setLength(o, s, l)
    }
}(),
n.ArrowHelper.prototype = Object.create(n.Object3D.prototype),
n.ArrowHelper.prototype.constructor = n.ArrowHelper,
n.ArrowHelper.prototype.setDirection = function() {
    var e, t = new n.Vector3;
    return function(i) {
        i.y > .99999 ? this.quaternion.set(0, 0, 0, 1) : i.y < -.99999 ? this.quaternion.set(1, 0, 0, 0) : (t.set(i.z, 0, -i.x).normalize(),
        e = Math.acos(i.y),
        this.quaternion.setFromAxisAngle(t, e))
    }
}(),
n.ArrowHelper.prototype.setLength = function(e, t, i) {
    void 0 === t && (t = .2 * e),
    void 0 === i && (i = .2 * t),
    this.line.scale.set(1, Math.max(0, e - t), 1),
    this.line.updateMatrix(),
    this.cone.scale.set(i, t, i),
    this.cone.position.y = e,
    this.cone.updateMatrix()
}
,
n.ArrowHelper.prototype.setColor = function(e) {
    this.line.material.color.set(e),
    this.cone.material.color.set(e)
}
,
n.BoxHelper = function(e) {
    var t = new Uint16Array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7])
      , i = new Float32Array(24)
      , r = new n.BufferGeometry;
    r.setIndex(new n.BufferAttribute(t,1)),
    r.addAttribute("position", new n.BufferAttribute(i,3)),
    n.LineSegments.call(this, r, new n.LineBasicMaterial({
        color: 16776960
    })),
    void 0 !== e && this.update(e)
}
,
n.BoxHelper.prototype = Object.create(n.LineSegments.prototype),
n.BoxHelper.prototype.constructor = n.BoxHelper,
n.BoxHelper.prototype.update = function() {
    var e = new n.Box3;
    return function(t) {
        if (e.setFromObject(t),
        !e.isEmpty()) {
            var i = e.min
              , n = e.max
              , r = this.geometry.attributes.position
              , o = r.array;
            o[0] = n.x,
            o[1] = n.y,
            o[2] = n.z,
            o[3] = i.x,
            o[4] = n.y,
            o[5] = n.z,
            o[6] = i.x,
            o[7] = i.y,
            o[8] = n.z,
            o[9] = n.x,
            o[10] = i.y,
            o[11] = n.z,
            o[12] = n.x,
            o[13] = n.y,
            o[14] = i.z,
            o[15] = i.x,
            o[16] = n.y,
            o[17] = i.z,
            o[18] = i.x,
            o[19] = i.y,
            o[20] = i.z,
            o[21] = n.x,
            o[22] = i.y,
            o[23] = i.z,
            r.needsUpdate = !0,
            this.geometry.computeBoundingSphere()
        }
    }
}(),
n.BoundingBoxHelper = function(e, t) {
    var i = void 0 !== t ? t : 8947848;
    this.object = e,
    this.box = new n.Box3,
    n.Mesh.call(this, new n.BoxGeometry(1,1,1), new n.MeshBasicMaterial({
        color: i,
        wireframe: !0
    }))
}
,
n.BoundingBoxHelper.prototype = Object.create(n.Mesh.prototype),
n.BoundingBoxHelper.prototype.constructor = n.BoundingBoxHelper,
n.BoundingBoxHelper.prototype.update = function() {
    this.box.setFromObject(this.object),
    this.box.size(this.scale),
    this.box.center(this.position)
}
,
n.CameraHelper = function(e) {
    function t(e, t, n) {
        i(e, n),
        i(t, n)
    }
    function i(e, t) {
        r.vertices.push(new n.Vector3),
        r.colors.push(new n.Color(t)),
        void 0 === a[e] && (a[e] = []),
        a[e].push(r.vertices.length - 1)
    }
    var r = new n.Geometry
      , o = new n.LineBasicMaterial({
        color: 16777215,
        vertexColors: n.FaceColors
    })
      , a = {}
      , s = 16755200
      , l = 16711680
      , h = 43775
      , c = 16777215
      , u = 3355443;
    t("n1", "n2", s),
    t("n2", "n4", s),
    t("n4", "n3", s),
    t("n3", "n1", s),
    t("f1", "f2", s),
    t("f2", "f4", s),
    t("f4", "f3", s),
    t("f3", "f1", s),
    t("n1", "f1", s),
    t("n2", "f2", s),
    t("n3", "f3", s),
    t("n4", "f4", s),
    t("p", "n1", l),
    t("p", "n2", l),
    t("p", "n3", l),
    t("p", "n4", l),
    t("u1", "u2", h),
    t("u2", "u3", h),
    t("u3", "u1", h),
    t("c", "t", c),
    t("p", "c", u),
    t("cn1", "cn2", u),
    t("cn3", "cn4", u),
    t("cf1", "cf2", u),
    t("cf3", "cf4", u),
    n.LineSegments.call(this, r, o),
    this.camera = e,
    this.camera.updateProjectionMatrix(),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1,
    this.pointMap = a,
    this.update()
}
,
n.CameraHelper.prototype = Object.create(n.LineSegments.prototype),
n.CameraHelper.prototype.constructor = n.CameraHelper,
n.CameraHelper.prototype.update = function() {
    function e(e, n, a, s) {
        r.set(n, a, s).unproject(o);
        var l = i[e];
        if (void 0 !== l)
            for (var h = 0, c = l.length; h < c; h++)
                t.vertices[l[h]].copy(r)
    }
    var t, i, r = new n.Vector3, o = new n.Camera;
    return function() {
        t = this.geometry,
        i = this.pointMap;
        var n = 1
          , r = 1;
        o.projectionMatrix.copy(this.camera.projectionMatrix),
        e("c", 0, 0, -1),
        e("t", 0, 0, 1),
        e("n1", -n, -r, -1),
        e("n2", n, -r, -1),
        e("n3", -n, r, -1),
        e("n4", n, r, -1),
        e("f1", -n, -r, 1),
        e("f2", n, -r, 1),
        e("f3", -n, r, 1),
        e("f4", n, r, 1),
        e("u1", .7 * n, 1.1 * r, -1),
        e("u2", .7 * -n, 1.1 * r, -1),
        e("u3", 0, 2 * r, -1),
        e("cf1", -n, 0, 1),
        e("cf2", n, 0, 1),
        e("cf3", 0, -r, 1),
        e("cf4", 0, r, 1),
        e("cn1", -n, 0, -1),
        e("cn2", n, 0, -1),
        e("cn3", 0, -r, -1),
        e("cn4", 0, r, -1),
        t.verticesNeedUpdate = !0
    }
}(),
n.DirectionalLightHelper = function(e, t) {
    n.Object3D.call(this),
    this.light = e,
    this.light.updateMatrixWorld(),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1,
    t = t || 1;
    var i = new n.Geometry;
    i.vertices.push(new n.Vector3(-t,t,0), new n.Vector3(t,t,0), new n.Vector3(t,-t,0), new n.Vector3(-t,-t,0), new n.Vector3(-t,t,0));
    var r = new n.LineBasicMaterial({
        fog: !1
    });
    r.color.copy(this.light.color).multiplyScalar(this.light.intensity),
    this.lightPlane = new n.Line(i,r),
    this.add(this.lightPlane),
    i = new n.Geometry,
    i.vertices.push(new n.Vector3, new n.Vector3),
    r = new n.LineBasicMaterial({
        fog: !1
    }),
    r.color.copy(this.light.color).multiplyScalar(this.light.intensity),
    this.targetLine = new n.Line(i,r),
    this.add(this.targetLine),
    this.update()
}
,
n.DirectionalLightHelper.prototype = Object.create(n.Object3D.prototype),
n.DirectionalLightHelper.prototype.constructor = n.DirectionalLightHelper,
n.DirectionalLightHelper.prototype.dispose = function() {
    this.lightPlane.geometry.dispose(),
    this.lightPlane.material.dispose(),
    this.targetLine.geometry.dispose(),
    this.targetLine.material.dispose()
}
,
n.DirectionalLightHelper.prototype.update = function() {
    var e = new n.Vector3
      , t = new n.Vector3
      , i = new n.Vector3;
    return function() {
        e.setFromMatrixPosition(this.light.matrixWorld),
        t.setFromMatrixPosition(this.light.target.matrixWorld),
        i.subVectors(t, e),
        this.lightPlane.lookAt(i),
        this.lightPlane.material.color.copy(this.light.color).multiplyScalar(this.light.intensity),
        this.targetLine.geometry.vertices[1].copy(i),
        this.targetLine.geometry.verticesNeedUpdate = !0,
        this.targetLine.material.color.copy(this.lightPlane.material.color)
    }
}(),
n.EdgesHelper = function(e, t, i) {
    var r = void 0 !== t ? t : 16777215;
    n.LineSegments.call(this, new n.EdgesGeometry(e.geometry,i), new n.LineBasicMaterial({
        color: r
    })),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1
}
,
n.EdgesHelper.prototype = Object.create(n.LineSegments.prototype),
n.EdgesHelper.prototype.constructor = n.EdgesHelper,
n.FaceNormalsHelper = function(e, t, i, r) {
    this.object = e,
    this.size = void 0 !== t ? t : 1;
    var o = void 0 !== i ? i : 16776960
      , a = void 0 !== r ? r : 1
      , s = 0
      , l = this.object.geometry;
    l instanceof n.Geometry ? s = l.faces.length : console.warn("THREE.FaceNormalsHelper: only THREE.Geometry is supported. Use THREE.VertexNormalsHelper, instead.");
    var h = new n.BufferGeometry
      , c = new n.Float32Attribute(2 * s * 3,3);
    h.addAttribute("position", c),
    n.LineSegments.call(this, h, new n.LineBasicMaterial({
        color: o,
        linewidth: a
    })),
    this.matrixAutoUpdate = !1,
    this.update()
}
,
n.FaceNormalsHelper.prototype = Object.create(n.LineSegments.prototype),
n.FaceNormalsHelper.prototype.constructor = n.FaceNormalsHelper,
n.FaceNormalsHelper.prototype.update = function() {
    var e = new n.Vector3
      , t = new n.Vector3
      , i = new n.Matrix3;
    return function() {
        this.object.updateMatrixWorld(!0),
        i.getNormalMatrix(this.object.matrixWorld);
        for (var n = this.object.matrixWorld, r = this.geometry.attributes.position, o = this.object.geometry, a = o.vertices, s = o.faces, l = 0, h = 0, c = s.length; h < c; h++) {
            var u = s[h]
              , d = u.normal;
            e.copy(a[u.a]).add(a[u.b]).add(a[u.c]).divideScalar(3).applyMatrix4(n),
            t.copy(d).applyMatrix3(i).normalize().multiplyScalar(this.size).add(e),
            r.setXYZ(l, e.x, e.y, e.z),
            l += 1,
            r.setXYZ(l, t.x, t.y, t.z),
            l += 1
        }
        return r.needsUpdate = !0,
        this
    }
}(),
n.GridHelper = function(e, t) {
    var i = new n.Geometry
      , r = new n.LineBasicMaterial({
        vertexColors: n.VertexColors
    });
    this.color1 = new n.Color(4473924),
    this.color2 = new n.Color(8947848);
    for (var o = -e; o <= e; o += t) {
        i.vertices.push(new n.Vector3(-e,0,o), new n.Vector3(e,0,o), new n.Vector3(o,0,-e), new n.Vector3(o,0,e));
        var a = 0 === o ? this.color1 : this.color2;
        i.colors.push(a, a, a, a)
    }
    n.LineSegments.call(this, i, r)
}
,
n.GridHelper.prototype = Object.create(n.LineSegments.prototype),
n.GridHelper.prototype.constructor = n.GridHelper,
n.GridHelper.prototype.setColors = function(e, t) {
    this.color1.set(e),
    this.color2.set(t),
    this.geometry.colorsNeedUpdate = !0
}
,
n.HemisphereLightHelper = function(e, t) {
    n.Object3D.call(this),
    this.light = e,
    this.light.updateMatrixWorld(),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1,
    this.colors = [new n.Color, new n.Color];
    var i = new n.SphereGeometry(t,4,2);
    i.rotateX(-Math.PI / 2);
    for (var r = 0, o = 8; r < o; r++)
        i.faces[r].color = this.colors[r < 4 ? 0 : 1];
    var a = new n.MeshBasicMaterial({
        vertexColors: n.FaceColors,
        wireframe: !0
    });
    this.lightSphere = new n.Mesh(i,a),
    this.add(this.lightSphere),
    this.update()
}
,
n.HemisphereLightHelper.prototype = Object.create(n.Object3D.prototype),
n.HemisphereLightHelper.prototype.constructor = n.HemisphereLightHelper,
n.HemisphereLightHelper.prototype.dispose = function() {
    this.lightSphere.geometry.dispose(),
    this.lightSphere.material.dispose()
}
,
n.HemisphereLightHelper.prototype.update = function() {
    var e = new n.Vector3;
    return function() {
        this.colors[0].copy(this.light.color).multiplyScalar(this.light.intensity),
        this.colors[1].copy(this.light.groundColor).multiplyScalar(this.light.intensity),
        this.lightSphere.lookAt(e.setFromMatrixPosition(this.light.matrixWorld).negate()),
        this.lightSphere.geometry.colorsNeedUpdate = !0
    }
}(),
n.PointLightHelper = function(e, t) {
    this.light = e,
    this.light.updateMatrixWorld();
    var i = new n.SphereGeometry(t,4,2)
      , r = new n.MeshBasicMaterial({
        wireframe: !0,
        fog: !1
    });
    r.color.copy(this.light.color).multiplyScalar(this.light.intensity),
    n.Mesh.call(this, i, r),
    this.matrix = this.light.matrixWorld,
    this.matrixAutoUpdate = !1
}
,
n.PointLightHelper.prototype = Object.create(n.Mesh.prototype),
n.PointLightHelper.prototype.constructor = n.PointLightHelper,
n.PointLightHelper.prototype.dispose = function() {
    this.geometry.dispose(),
    this.material.dispose()
}
,
n.PointLightHelper.prototype.update = function() {
    this.material.color.copy(this.light.color).multiplyScalar(this.light.intensity)
}
,
n.SkeletonHelper = function(e) {
    this.bones = this.getBoneList(e);
    for (var t = new n.Geometry, i = 0; i < this.bones.length; i++) {
        var r = this.bones[i];
        r.parent instanceof n.Bone && (t.vertices.push(new n.Vector3),
        t.vertices.push(new n.Vector3),
        t.colors.push(new n.Color(0,0,1)),
        t.colors.push(new n.Color(0,1,0)))
    }
    t.dynamic = !0;
    var o = new n.LineBasicMaterial({
        vertexColors: n.VertexColors,
        depthTest: !1,
        depthWrite: !1,
        transparent: !0
    });
    n.LineSegments.call(this, t, o),
    this.root = e,
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1,
    this.update()
}
,
n.SkeletonHelper.prototype = Object.create(n.LineSegments.prototype),
n.SkeletonHelper.prototype.constructor = n.SkeletonHelper,
n.SkeletonHelper.prototype.getBoneList = function(e) {
    var t = [];
    e instanceof n.Bone && t.push(e);
    for (var i = 0; i < e.children.length; i++)
        t.push.apply(t, this.getBoneList(e.children[i]));
    return t
}
,
n.SkeletonHelper.prototype.update = function() {
    for (var e = this.geometry, t = (new n.Matrix4).getInverse(this.root.matrixWorld), i = new n.Matrix4, r = 0, o = 0; o < this.bones.length; o++) {
        var a = this.bones[o];
        a.parent instanceof n.Bone && (i.multiplyMatrices(t, a.matrixWorld),
        e.vertices[r].setFromMatrixPosition(i),
        i.multiplyMatrices(t, a.parent.matrixWorld),
        e.vertices[r + 1].setFromMatrixPosition(i),
        r += 2)
    }
    e.verticesNeedUpdate = !0,
    e.computeBoundingSphere()
}
,
n.SpotLightHelper = function(e) {
    n.Object3D.call(this),
    this.light = e,
    this.light.updateMatrixWorld(),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1;
    var t = new n.CylinderGeometry(0,1,1,8,1,!0);
    t.translate(0, -.5, 0),
    t.rotateX(-Math.PI / 2);
    var i = new n.MeshBasicMaterial({
        wireframe: !0,
        fog: !1
    });
    this.cone = new n.Mesh(t,i),
    this.add(this.cone),
    this.update()
}
,
n.SpotLightHelper.prototype = Object.create(n.Object3D.prototype),
n.SpotLightHelper.prototype.constructor = n.SpotLightHelper,
n.SpotLightHelper.prototype.dispose = function() {
    this.cone.geometry.dispose(),
    this.cone.material.dispose()
}
,
n.SpotLightHelper.prototype.update = function() {
    var e = new n.Vector3
      , t = new n.Vector3;
    return function() {
        var i = this.light.distance ? this.light.distance : 1e4
          , n = i * Math.tan(this.light.angle);
        this.cone.scale.set(n, n, i),
        e.setFromMatrixPosition(this.light.matrixWorld),
        t.setFromMatrixPosition(this.light.target.matrixWorld),
        this.cone.lookAt(t.sub(e)),
        this.cone.material.color.copy(this.light.color).multiplyScalar(this.light.intensity)
    }
}(),
n.VertexNormalsHelper = function(e, t, i, r) {
    this.object = e,
    this.size = void 0 !== t ? t : 1;
    var o = void 0 !== i ? i : 16711680
      , a = void 0 !== r ? r : 1
      , s = 0
      , l = this.object.geometry;
    l instanceof n.Geometry ? s = 3 * l.faces.length : l instanceof n.BufferGeometry && (s = l.attributes.normal.count);
    var h = new n.BufferGeometry
      , c = new n.Float32Attribute(2 * s * 3,3);
    h.addAttribute("position", c),
    n.LineSegments.call(this, h, new n.LineBasicMaterial({
        color: o,
        linewidth: a
    })),
    this.matrixAutoUpdate = !1,
    this.update()
}
,
n.VertexNormalsHelper.prototype = Object.create(n.LineSegments.prototype),
n.VertexNormalsHelper.prototype.constructor = n.VertexNormalsHelper,
n.VertexNormalsHelper.prototype.update = function() {
    var e = new n.Vector3
      , t = new n.Vector3
      , i = new n.Matrix3;
    return function() {
        var r = ["a", "b", "c"];
        this.object.updateMatrixWorld(!0),
        i.getNormalMatrix(this.object.matrixWorld);
        var o = this.object.matrixWorld
          , a = this.geometry.attributes.position
          , s = this.object.geometry;
        if (s instanceof n.Geometry)
            for (var l = s.vertices, h = s.faces, c = 0, u = 0, d = h.length; u < d; u++)
                for (var p = h[u], f = 0, g = p.vertexNormals.length; f < g; f++) {
                    var m = l[p[r[f]]]
                      , v = p.vertexNormals[f];
                    e.copy(m).applyMatrix4(o),
                    t.copy(v).applyMatrix3(i).normalize().multiplyScalar(this.size).add(e),
                    a.setXYZ(c, e.x, e.y, e.z),
                    c += 1,
                    a.setXYZ(c, t.x, t.y, t.z),
                    c += 1
                }
        else if (s instanceof n.BufferGeometry)
            for (var A = s.attributes.position, y = s.attributes.normal, c = 0, f = 0, g = A.count; f < g; f++)
                e.set(A.getX(f), A.getY(f), A.getZ(f)).applyMatrix4(o),
                t.set(y.getX(f), y.getY(f), y.getZ(f)),
                t.applyMatrix3(i).normalize().multiplyScalar(this.size).add(e),
                a.setXYZ(c, e.x, e.y, e.z),
                c += 1,
                a.setXYZ(c, t.x, t.y, t.z),
                c += 1;
        return a.needsUpdate = !0,
        this
    }
}(),
n.WireframeHelper = function(e, t) {
    var i = void 0 !== t ? t : 16777215;
    n.LineSegments.call(this, new n.WireframeGeometry(e.geometry), new n.LineBasicMaterial({
        color: i
    })),
    this.matrix = e.matrixWorld,
    this.matrixAutoUpdate = !1
}
,
n.WireframeHelper.prototype = Object.create(n.LineSegments.prototype),
n.WireframeHelper.prototype.constructor = n.WireframeHelper,
n.ImmediateRenderObject = function(e) {
    n.Object3D.call(this),
    this.material = e,
    this.render = function(e) {}
}
,
n.ImmediateRenderObject.prototype = Object.create(n.Object3D.prototype),
n.ImmediateRenderObject.prototype.constructor = n.ImmediateRenderObject,
n.MorphBlendMesh = function(e, t) {
    n.Mesh.call(this, e, t),
    this.animationsMap = {},
    this.animationsList = [];
    var i = this.geometry.morphTargets.length
      , r = "__default"
      , o = 0
      , a = i - 1
      , s = i / 1;
    this.createAnimation(r, o, a, s),
    this.setAnimationWeight(r, 1)
}
,
n.MorphBlendMesh.prototype = Object.create(n.Mesh.prototype),
n.MorphBlendMesh.prototype.constructor = n.MorphBlendMesh,
n.MorphBlendMesh.prototype.createAnimation = function(e, t, i, n) {
    var r = {
        start: t,
        end: i,
        length: i - t + 1,
        fps: n,
        duration: (i - t) / n,
        lastFrame: 0,
        currentFrame: 0,
        active: !1,
        time: 0,
        direction: 1,
        weight: 1,
        directionBackwards: !1,
        mirroredLoop: !1
    };
    this.animationsMap[e] = r,
    this.animationsList.push(r)
}
,
n.MorphBlendMesh.prototype.autoCreateAnimations = function(e) {
    for (var t, i = /([a-z]+)_?(\d+)/i, n = {}, r = this.geometry, o = 0, a = r.morphTargets.length; o < a; o++) {
        var s = r.morphTargets[o]
          , l = s.name.match(i);
        if (l && l.length > 1) {
            var h = l[1];
            n[h] || (n[h] = {
                start: 1 / 0,
                end: -(1 / 0)
            });
            var c = n[h];
            o < c.start && (c.start = o),
            o > c.end && (c.end = o),
            t || (t = h)
        }
    }
    for (var h in n) {
        var c = n[h];
        this.createAnimation(h, c.start, c.end, e)
    }
    this.firstAnimation = t
}
,
n.MorphBlendMesh.prototype.setAnimationDirectionForward = function(e) {
    var t = this.animationsMap[e];
    t && (t.direction = 1,
    t.directionBackwards = !1)
}
,
n.MorphBlendMesh.prototype.setAnimationDirectionBackward = function(e) {
    var t = this.animationsMap[e];
    t && (t.direction = -1,
    t.directionBackwards = !0)
}
,
n.MorphBlendMesh.prototype.setAnimationFPS = function(e, t) {
    var i = this.animationsMap[e];
    i && (i.fps = t,
    i.duration = (i.end - i.start) / i.fps)
}
,
n.MorphBlendMesh.prototype.setAnimationDuration = function(e, t) {
    var i = this.animationsMap[e];
    i && (i.duration = t,
    i.fps = (i.end - i.start) / i.duration)
}
,
n.MorphBlendMesh.prototype.setAnimationWeight = function(e, t) {
    var i = this.animationsMap[e];
    i && (i.weight = t)
}
,
n.MorphBlendMesh.prototype.setAnimationTime = function(e, t) {
    var i = this.animationsMap[e];
    i && (i.time = t)
}
,
n.MorphBlendMesh.prototype.getAnimationTime = function(e) {
    var t = 0
      , i = this.animationsMap[e];
    return i && (t = i.time),
    t
}
,
n.MorphBlendMesh.prototype.getAnimationDuration = function(e) {
    var t = -1
      , i = this.animationsMap[e];
    return i && (t = i.duration),
    t
}
,
n.MorphBlendMesh.prototype.playAnimation = function(e) {
    var t = this.animationsMap[e];
    t ? (t.time = 0,
    t.active = !0) : console.warn("THREE.MorphBlendMesh: animation[" + e + "] undefined in .playAnimation()")
}
,
n.MorphBlendMesh.prototype.stopAnimation = function(e) {
    var t = this.animationsMap[e];
    t && (t.active = !1)
}
,
n.MorphBlendMesh.prototype.update = function(e) {
    for (var t = 0, i = this.animationsList.length; t < i; t++) {
        var r = this.animationsList[t];
        if (r.active) {
            var o = r.duration / r.length;
            r.time += r.direction * e,
            r.mirroredLoop ? (r.time > r.duration || r.time < 0) && (r.direction *= -1,
            r.time > r.duration && (r.time = r.duration,
            r.directionBackwards = !0),
            r.time < 0 && (r.time = 0,
            r.directionBackwards = !1)) : (r.time = r.time % r.duration,
            r.time < 0 && (r.time += r.duration));
            var a = r.start + n.Math.clamp(Math.floor(r.time / o), 0, r.length - 1)
              , s = r.weight;
            a !== r.currentFrame && (this.morphTargetInfluences[r.lastFrame] = 0,
            this.morphTargetInfluences[r.currentFrame] = 1 * s,
            this.morphTargetInfluences[a] = 0,
            r.lastFrame = r.currentFrame,
            r.currentFrame = a);
            var l = r.time % o / o;
            r.directionBackwards && (l = 1 - l),
            r.currentFrame !== r.lastFrame ? (this.morphTargetInfluences[r.currentFrame] = l * s,
            this.morphTargetInfluences[r.lastFrame] = (1 - l) * s) : this.morphTargetInfluences[r.currentFrame] = s
        }
    }
};

/*
3D Model Scraping: 
Hepler function for converting threejs to obj 
*/
THREE = n;
n.OBJExporter = function () {};
n.OBJExporter.prototype = {
	constructor: THREE.OBJExporter,
	parse: function ( object ) {
		var output = '';

		var indexVertex = 0;
		var indexVertexUvs = 0;
		var indexNormals = 0;

		var vertex = new THREE.Vector3();
		var normal = new THREE.Vector3();
		var uv = new THREE.Vector2();

		var i, j, l, m, face = [];

		var parseMesh = function ( mesh ) {

			var nbVertex = 0;
			var nbNormals = 0;
			var nbVertexUvs = 0;

			var geometry = mesh.geometry;

			var normalMatrixWorld = new THREE.Matrix3();

			if ( geometry instanceof THREE.Geometry ) {

				geometry = new THREE.BufferGeometry().setFromObject( mesh );

			}

			if ( geometry instanceof THREE.BufferGeometry ) {

				// shortcuts
				var vertices = geometry.getAttribute( 'position' );
				var normals = geometry.getAttribute( 'normal' );
				var uvs = geometry.getAttribute( 'uv' );
				var indices = geometry.getIndex();

				// name of the mesh object
				output += 'o ' + mesh.name + '\n';

				// vertices

				if( vertices !== undefined ) {

					for ( i = 0, l = vertices.count; i < l; i ++, nbVertex++ ) {

						vertex.x = vertices.getX( i );
						vertex.y = vertices.getY( i );
						vertex.z = vertices.getZ( i );

						// transfrom the vertex to world space
						vertex.applyMatrix4( mesh.matrixWorld );

						// transform the vertex to export format
						output += 'v ' + vertex.x + ' ' + vertex.y + ' ' + vertex.z + '\n';

					}

				}

				// uvs

				if( uvs !== undefined ) {

					for ( i = 0, l = uvs.count; i < l; i ++, nbVertexUvs++ ) {

						uv.x = uvs.getX( i );
						uv.y = uvs.getY( i );

						// transform the uv to export format
						output += 'vt ' + uv.x + ' ' + uv.y + '\n';

					}

				}

				// normals

				if( normals !== undefined ) {

					normalMatrixWorld.getNormalMatrix( mesh.matrixWorld );

					for ( i = 0, l = normals.count; i < l; i ++, nbNormals++ ) {

						normal.x = normals.getX( i );
						normal.y = normals.getY( i );
						normal.z = normals.getZ( i );

						// transfrom the normal to world space
						normal.applyMatrix3( normalMatrixWorld );

						// transform the normal to export format
						output += 'vn ' + normal.x + ' ' + normal.y + ' ' + normal.z + '\n';

					}

				}

				// faces

				if( indices !== null ) {

					for ( i = 0, l = indices.count; i < l; i += 3 ) {

						for( m = 0; m < 3; m ++ ){

							j = indices.getX( i + m ) + 1;

							face[ m ] = ( indexVertex + j ) + '/' + ( uvs ? ( indexVertexUvs + j ) : '' ) + '/' + ( indexNormals + j );

						}

						// transform the face to export format
						output += 'f ' + face.join( ' ' ) + "\n";

					}

				} else {

					for ( i = 0, l = vertices.count; i < l; i += 3 ) {

						for( m = 0; m < 3; m ++ ){

							j = i + m + 1;

							face[ m ] = ( indexVertex + j ) + '/' + ( uvs ? ( indexVertexUvs + j ) : '' ) + '/' + ( indexNormals + j );

						}

						// transform the face to export format
						output += 'f ' + face.join( ' ' ) + "\n";

					}

				}

			} else {

				console.warn( 'THREE.OBJExporter.parseMesh(): geometry type unsupported', geometry );

			}

			// update index
			indexVertex += nbVertex;
			indexVertexUvs += nbVertexUvs;
			indexNormals += nbNormals;

		};

		var parseLine = function( line ) {

			var nbVertex = 0;

			var geometry = line.geometry;
			var type = line.type;

			if ( geometry instanceof THREE.Geometry ) {

				geometry = new THREE.BufferGeometry().setFromObject( line );

			}

			if ( geometry instanceof THREE.BufferGeometry ) {

				// shortcuts
				var vertices = geometry.getAttribute( 'position' );
				var indices = geometry.getIndex();

				// name of the line object
				output += 'o ' + line.name + '\n';

				if( vertices !== undefined ) {

					for ( i = 0, l = vertices.count; i < l; i ++, nbVertex++ ) {

						vertex.x = vertices.getX( i );
						vertex.y = vertices.getY( i );
						vertex.z = vertices.getZ( i );

						// transfrom the vertex to world space
						vertex.applyMatrix4( line.matrixWorld );

						// transform the vertex to export format
						output += 'v ' + vertex.x + ' ' + vertex.y + ' ' + vertex.z + '\n';

					}

				}

				if ( type === 'Line' ) {

					output += 'l ';

					for ( j = 1, l = vertices.count; j <= l; j++ ) {

						output += ( indexVertex + j ) + ' ';

					}

					output += '\n';

				}

				if ( type === 'LineSegments' ) {

					for ( j = 1, k = j + 1, l = vertices.count; j < l; j += 2, k = j + 1 ) {

						output += 'l ' + ( indexVertex + j ) + ' ' + ( indexVertex + k ) + '\n';

					}

				}

			} else {

				console.warn('THREE.OBJExporter.parseLine(): geometry type unsupported', geometry );

			}

			// update index
			indexVertex += nbVertex;

		};

		object.traverse = function ( callback ) {

			callback( object );

			var children = object.children;

			for ( var i = 0, l = children.length; i < l; i ++ ) {

				children[ i ].traverse( callback );

			}

		},

		object.traverse( function ( child ) {

			if ( child instanceof THREE.Mesh ) {

				parseMesh( child );

			}

			if ( child instanceof THREE.Line ) {

				parseLine( child );

			}

		} );

		return output;

	}

};

/* 
3D Model Scraping:
Helper function for merging multiple 3D elements
*/
n.BufferGeometry.prototype.merge = function ( geometry ) {

    if ( geometry instanceof THREE.BufferGeometry === false ) {

        console.error( 'THREE.BufferGeometry.merge(): geometry not an instance of THREE.BufferGeometry.', geometry );
        return;

    }

    var attributes = this.attributes;

    if( this.index ){

        var indices = geometry.index.array;

        var offset = attributes[ 'position' ].count;

        for( var i = 0, il = indices.length; i < il; i++ ) {

            indices[i] = offset + indices[i];

        }

        this.index.array = Uint32ArrayConcat( this.index.array, indices );

    }

    for ( var key in attributes ) {

        if ( geometry.attributes[ key ] === undefined ) continue;

        attributes[ key ].array = Float32ArrayConcat( attributes[ key ].array, geometry.attributes[ key ].array );

    }

    return this;

    /***
     * @param {Float32Array} first
     * @param {Float32Array} second
     * @returns {Float32Array}
     * @constructor
     */
    function Float32ArrayConcat(first, second)
    {
        var firstLength = first.length,
            result = new Float32Array(firstLength + second.length);

        result.set(first);
        result.set(second, firstLength);

        return result;
    }

    /**
     * @param {Uint32Array} first
     * @param {Uint32Array} second
     * @returns {Uint32Array}
     * @constructor
     */
    function Uint32ArrayConcat(first, second)
    {
        var firstLength = first.length,
            result = new Uint32Array(firstLength + second.length);

        result.set(first);
        result.set(second, firstLength);

        return result;
    }

};



module.exports = THREE