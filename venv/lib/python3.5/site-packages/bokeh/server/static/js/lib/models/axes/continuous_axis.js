"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var axis_1 = require("./axis");
var ContinuousAxis = /** @class */ (function (_super) {
    tslib_1.__extends(ContinuousAxis, _super);
    function ContinuousAxis(attrs) {
        return _super.call(this, attrs) || this;
    }
    ContinuousAxis.__name__ = "ContinuousAxis";
    return ContinuousAxis;
}(axis_1.Axis));
exports.ContinuousAxis = ContinuousAxis;
