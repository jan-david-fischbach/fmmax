"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3197],{3905:(t,e,r)=>{r.d(e,{Zo:()=>p,kt:()=>d});var a=r(7294);function n(t,e,r){return e in t?Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}):t[e]=r,t}function o(t,e){var r=Object.keys(t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(t);e&&(a=a.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),r.push.apply(r,a)}return r}function i(t){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{};e%2?o(Object(r),!0).forEach((function(e){n(t,e,r[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(r,e))}))}return t}function l(t,e){if(null==t)return{};var r,a,n=function(t,e){if(null==t)return{};var r,a,n={},o=Object.keys(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||(n[r]=t[r]);return n}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(t,r)&&(n[r]=t[r])}return n}var s=a.createContext({}),m=function(t){var e=a.useContext(s),r=e;return t&&(r="function"==typeof t?t(e):i(i({},e),t)),r},p=function(t){var e=m(t.components);return a.createElement(s.Provider,{value:e},t.children)},u="mdxType",c={inlineCode:"code",wrapper:function(t){var e=t.children;return a.createElement(a.Fragment,{},e)}},b=a.forwardRef((function(t,e){var r=t.components,n=t.mdxType,o=t.originalType,s=t.parentName,p=l(t,["components","mdxType","originalType","parentName"]),u=m(r),b=n,d=u["".concat(s,".").concat(b)]||u[b]||c[b]||o;return r?a.createElement(d,i(i({ref:e},p),{},{components:r})):a.createElement(d,i({ref:e},p))}));function d(t,e){var r=arguments,n=e&&e.mdxType;if("string"==typeof t||n){var o=r.length,i=new Array(o);i[0]=b;var l={};for(var s in e)hasOwnProperty.call(e,s)&&(l[s]=e[s]);l.originalType=t,l[u]="string"==typeof t?t:n,i[1]=l;for(var m=2;m<o;m++)i[m]=r[m];return a.createElement.apply(null,i)}return a.createElement.apply(null,r)}b.displayName="MDXCreateElement"},1739:(t,e,r)=>{r.r(e),r.d(e,{assets:()=>s,contentTitle:()=>i,default:()=>c,frontMatter:()=>o,metadata:()=>l,toc:()=>m});var a=r(7462),n=(r(7294),r(3905));const o={id:"beams.rotation_matrix"},i=void 0,l={unversionedId:"API/beams.rotation_matrix",id:"API/beams.rotation_matrix",title:"beams.rotation_matrix",description:"beams.rotation_matrix",source:"@site/docs/API/beams.rotation_matrix.md",sourceDirName:"API",slug:"/API/beams.rotation_matrix",permalink:"/fmmax/API/beams.rotation_matrix",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/beams.rotation_matrix.md",tags:[],version:"current",frontMatter:{id:"beams.rotation_matrix"},sidebar:"APISidebar",previous:{title:"basis.unit_cell_coordinates",permalink:"/fmmax/API/basis.unit_cell_coordinates"},next:{title:"beams.shifted_rotated_fields",permalink:"/fmmax/API/beams.shifted_rotated_fields"}},s={},m=[{value:"<code>beams.rotation_matrix</code>",id:"beamsrotation_matrix",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],p={toc:m},u="wrapper";function c(t){let{components:e,...r}=t;return(0,n.kt)(u,(0,a.Z)({},p,r,{components:e,mdxType:"MDXLayout"}),(0,n.kt)("h3",{id:"beamsrotation_matrix"},(0,n.kt)("inlineCode",{parentName:"h3"},"beams.rotation_matrix")),(0,n.kt)("p",null,"Computes a rotation matrix that arbitrarily reorients a field."),(0,n.kt)("p",null,"The rotation operations consist of,"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"rotation about the y-axis by ",(0,n.kt)("inlineCode",{parentName:"li"},"polar_angle")),(0,n.kt)("li",{parentName:"ul"},"rotation about the z-axis by ",(0,n.kt)("inlineCode",{parentName:"li"},"azimuthal_angle")),(0,n.kt)("li",{parentName:"ul"},"rotation about propagation axis by ",(0,n.kt)("inlineCode",{parentName:"li"},"polarization_angle"),", where\nthe propagation axis is found by applying the first two rotations\nabout the y- and z-axis.")),(0,n.kt)("h4",{id:"args"},"Args:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"polar_angle"),": The rotation angle about the y-axis."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"azimuthal_angle"),": The rotation angle about the z-axis."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"polarization_angle"),": The rotation angle about the propagation axis.")),(0,n.kt)("h4",{id:"returns"},"Returns:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"None"),": The rotation matrix.")))}c.isMDXComponent=!0}}]);