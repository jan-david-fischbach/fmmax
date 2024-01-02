"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[353],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>u});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),d=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=d(e.components);return r.createElement(s.Provider,{value:t},e.children)},f="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},c=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),f=d(n),c=a,u=f["".concat(s,".").concat(c)]||f[c]||m[c]||i;return n?r.createElement(u,o(o({ref:t},p),{},{components:n})):r.createElement(u,o({ref:t},p))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=c;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[f]="string"==typeof e?e:a,o[1]=l;for(var d=2;d<i;d++)o[d]=n[d];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}c.displayName="MDXCreateElement"},711:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>m,frontMatter:()=>i,metadata:()=>l,toc:()=>d});var r=n(7462),a=(n(7294),n(3905));const i={id:"beams.shifted_rotated_fields"},o=void 0,l={unversionedId:"API/beams.shifted_rotated_fields",id:"API/beams.shifted_rotated_fields",title:"beams.shifted_rotated_fields",description:"beams.shiftedrotatedfields",source:"@site/docs/API/beams.shifted_rotated_fields.md",sourceDirName:"API",slug:"/API/beams.shifted_rotated_fields",permalink:"/fmmax/API/beams.shifted_rotated_fields",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/beams.shifted_rotated_fields.md",tags:[],version:"current",frontMatter:{id:"beams.shifted_rotated_fields"},sidebar:"APISidebar",previous:{title:"beams.rotation_matrix",permalink:"/fmmax/API/beams.rotation_matrix"},next:{title:"farfield.angles_from_unflattened_transverse_wavevectors",permalink:"/fmmax/API/farfield.angles_from_unflattened_transverse_wavevectors"}},s={},d=[{value:"<code>beams.shifted_rotated_fields</code>",id:"beamsshifted_rotated_fields",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],p={toc:d},f="wrapper";function m(e){let{components:t,...n}=e;return(0,a.kt)(f,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h3",{id:"beamsshifted_rotated_fields"},(0,a.kt)("inlineCode",{parentName:"h3"},"beams.shifted_rotated_fields")),(0,a.kt)("p",null,"Computes the fields on a rotated coordinate system."),(0,a.kt)("p",null,"Given ",(0,a.kt)("inlineCode",{parentName:"p"},"fields_fn(xf, yf, zf) -> (exf, eyf, ezf), (hxf, hyf, hzf)")," which\nreturns the fields ",(0,a.kt)("em",{parentName:"p"},"in the field coordinate system"),", returns the fields\nat coordinates ",(0,a.kt)("inlineCode",{parentName:"p"},"(x, y, z)"),", which are rotated from ",(0,a.kt)("inlineCode",{parentName:"p"},"(xf, yf, zf)"),", by\nthe specified ",(0,a.kt)("inlineCode",{parentName:"p"},"polar_angle"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"azimuthal_angle"),", and ",(0,a.kt)("inlineCode",{parentName:"p"},"polarization_angle"),"."),(0,a.kt)("p",null,"A beam propagating in the ",(0,a.kt)("inlineCode",{parentName:"p"},"zf")," direction, polarized in the ",(0,a.kt)("inlineCode",{parentName:"p"},"xf")," direction\nwill be propagating in the direction specified by ",(0,a.kt)("inlineCode",{parentName:"p"},"polar_angle")," and\n",(0,a.kt)("inlineCode",{parentName:"p"},"azimuthal_angle"),", with polarization rotated about the propagation\ndirection by ",(0,a.kt)("inlineCode",{parentName:"p"},"polarization_angle"),"."),(0,a.kt)("h4",{id:"args"},"Args:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"field_fn"),": Function which returns the fields in the field coordinate\nsystem. The fields should be for a beam propagating in the zf\ndirection, i.e. in the z-direction of the beam coordinate system."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"x"),": x-coordinates of the desired output fields."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"y"),": y-coordinates of the desired output fields."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"z"),": z-coordinates of the desired output fields."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"beam_origin_x"),": The x-origin of the beam coordinate system in the\n",(0,a.kt)("inlineCode",{parentName:"li"},"(x, y, z)")," unit system."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"beam_origin_y"),": The y-origin of the beam coordinate system."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"beam_origin_z"),": The z-origin of the beam coordinate system."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"polar_angle"),": The rotation angle about the y-axis."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"azimuthal_angle"),": The rotation angle about the z-axis."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"polarization_angle"),": The rotation angle about the propagation axis.")),(0,a.kt)("h4",{id:"returns"},"Returns:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"None"),": The fields ",(0,a.kt)("inlineCode",{parentName:"li"},"((ex, ey, ez), (hx, hy, hz))")," at the specified coordinates.")))}m.isMDXComponent=!0}}]);