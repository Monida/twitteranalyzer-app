"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var FlatBush = require("flatbush");
var bbox_1 = require("./bbox");
var SpatialIndex = /** @class */ (function () {
    function SpatialIndex(points) {
        this.points = points;
        this.index = null;
        if (points.length > 0) {
            this.index = new FlatBush(points.length);
            for (var _i = 0, points_1 = points; _i < points_1.length; _i++) {
                var p = points_1[_i];
                var x0 = p.x0, y0 = p.y0, x1 = p.x1, y1 = p.y1;
                this.index.add(x0, y0, x1, y1);
            }
            this.index.finish();
        }
    }
    SpatialIndex.prototype._normalize = function (rect) {
        var _a, _b;
        var x0 = rect.x0, y0 = rect.y0, x1 = rect.x1, y1 = rect.y1;
        if (x0 > x1)
            _a = [x1, x0], x0 = _a[0], x1 = _a[1];
        if (y0 > y1)
            _b = [y1, y0], y0 = _b[0], y1 = _b[1];
        return { x0: x0, y0: y0, x1: x1, y1: y1 };
    };
    Object.defineProperty(SpatialIndex.prototype, "bbox", {
        get: function () {
            if (this.index == null)
                return bbox_1.empty();
            else {
                var _a = this.index, minX = _a.minX, minY = _a.minY, maxX = _a.maxX, maxY = _a.maxY;
                return { x0: minX, y0: minY, x1: maxX, y1: maxY };
            }
        },
        enumerable: true,
        configurable: true
    });
    SpatialIndex.prototype.search = function (rect) {
        var _this = this;
        if (this.index == null)
            return [];
        else {
            var _a = this._normalize(rect), x0 = _a.x0, y0 = _a.y0, x1 = _a.x1, y1 = _a.y1;
            var indices = this.index.search(x0, y0, x1, y1);
            return indices.map(function (j) { return _this.points[j]; });
        }
    };
    SpatialIndex.prototype.indices = function (rect) {
        return this.search(rect).map(function (_a) {
            var i = _a.i;
            return i;
        });
    };
    SpatialIndex.__name__ = "SpatialIndex";
    return SpatialIndex;
}());
exports.SpatialIndex = SpatialIndex;
