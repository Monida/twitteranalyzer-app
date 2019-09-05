"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var xy_glyph_1 = require("./xy_glyph");
var utils_1 = require("./utils");
var hittest = require("../../core/hittest");
var PatchView = /** @class */ (function (_super) {
    tslib_1.__extends(PatchView, _super);
    function PatchView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    PatchView.prototype._inner_loop = function (ctx, indices, sx, sy, func) {
        for (var _i = 0, indices_1 = indices; _i < indices_1.length; _i++) {
            var i = indices_1[_i];
            if (i == 0) {
                ctx.beginPath();
                ctx.moveTo(sx[i], sy[i]);
                continue;
            }
            else if (isNaN(sx[i] + sy[i])) {
                ctx.closePath();
                func.apply(ctx);
                ctx.beginPath();
                continue;
            }
            else
                ctx.lineTo(sx[i], sy[i]);
        }
        ctx.closePath();
        func.call(ctx);
    };
    PatchView.prototype._render = function (ctx, indices, _a) {
        var _this = this;
        var sx = _a.sx, sy = _a.sy;
        if (this.visuals.fill.doit) {
            this.visuals.fill.set_value(ctx);
            this._inner_loop(ctx, indices, sx, sy, ctx.fill);
        }
        this.visuals.hatch.doit2(ctx, 0, function () { return _this._inner_loop(ctx, indices, sx, sy, ctx.fill); }, function () { return _this.renderer.request_render(); });
        if (this.visuals.line.doit) {
            this.visuals.line.set_value(ctx);
            this._inner_loop(ctx, indices, sx, sy, ctx.stroke);
        }
    };
    PatchView.prototype.draw_legend_for_index = function (ctx, bbox, index) {
        utils_1.generic_area_legend(this.visuals, ctx, bbox, index);
    };
    PatchView.prototype._hit_point = function (geometry) {
        var _this = this;
        var result = hittest.create_empty_hit_test_result();
        if (hittest.point_in_poly(geometry.sx, geometry.sy, this.sx, this.sy)) {
            result.add_to_selected_glyphs(this.model);
            result.get_view = function () { return _this; };
        }
        return result;
    };
    PatchView.__name__ = "PatchView";
    return PatchView;
}(xy_glyph_1.XYGlyphView));
exports.PatchView = PatchView;
var Patch = /** @class */ (function (_super) {
    tslib_1.__extends(Patch, _super);
    function Patch(attrs) {
        return _super.call(this, attrs) || this;
    }
    Patch.initClass = function () {
        this.prototype.default_view = PatchView;
        this.mixins(['line', 'fill', 'hatch']);
    };
    Patch.__name__ = "Patch";
    return Patch;
}(xy_glyph_1.XYGlyph));
exports.Patch = Patch;
Patch.initClass();
