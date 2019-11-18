"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var box_1 = require("./box");
var QuadView = /** @class */ (function (_super) {
    tslib_1.__extends(QuadView, _super);
    function QuadView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    QuadView.prototype.scenterx = function (i) {
        return (this.sleft[i] + this.sright[i]) / 2;
    };
    QuadView.prototype.scentery = function (i) {
        return (this.stop[i] + this.sbottom[i]) / 2;
    };
    QuadView.prototype._index_data = function () {
        return this._index_box(this._right.length);
    };
    QuadView.prototype._lrtb = function (i) {
        var l = this._left[i];
        var r = this._right[i];
        var t = this._top[i];
        var b = this._bottom[i];
        return [l, r, t, b];
    };
    QuadView.__name__ = "QuadView";
    return QuadView;
}(box_1.BoxView));
exports.QuadView = QuadView;
var Quad = /** @class */ (function (_super) {
    tslib_1.__extends(Quad, _super);
    function Quad(attrs) {
        return _super.call(this, attrs) || this;
    }
    Quad.initClass = function () {
        this.prototype.default_view = QuadView;
        this.coords([['right', 'bottom'], ['left', 'top']]);
    };
    Quad.__name__ = "Quad";
    return Quad;
}(box_1.Box));
exports.Quad = Quad;
Quad.initClass();
