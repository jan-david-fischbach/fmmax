"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1278],{3905:(e,t,r)=>{r.d(t,{Zo:()=>p,kt:()=>s});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function c(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var l=n.createContext({}),u=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},p=function(e){var t=u(e.components);return n.createElement(l.Provider,{value:t},e.children)},d="mdxType",f={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,i=e.originalType,l=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),d=u(r),m=o,s=d["".concat(l,".").concat(m)]||d[m]||f[m]||i;return r?n.createElement(s,a(a({ref:t},p),{},{components:r})):n.createElement(s,a({ref:t},p))}));function s(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=r.length,a=new Array(i);a[0]=m;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c[d]="string"==typeof e?e:o,a[1]=c;for(var u=2;u<i;u++)a[u]=r[u];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},4282:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>a,default:()=>f,frontMatter:()=>i,metadata:()=>c,toc:()=>u});var n=r(7462),o=(r(7294),r(3905));const i={id:"vector_fourier.compute_gradient"},a=void 0,c={unversionedId:"API/vector_fourier.compute_gradient",id:"API/vector_fourier.compute_gradient",title:"vector_fourier.compute_gradient",description:"vectorfourier.computegradient",source:"@site/docs/API/vector_fourier.compute_gradient.md",sourceDirName:"API",slug:"/API/vector_fourier.compute_gradient",permalink:"/fmmax/API/vector_fourier.compute_gradient",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/vector_fourier.compute_gradient.md",tags:[],version:"current",frontMatter:{id:"vector_fourier.compute_gradient"},sidebar:"APISidebar",previous:{title:"vector_fourier.compute_field_pol",permalink:"/fmmax/API/vector_fourier.compute_field_pol"},next:{title:"vector_fourier.compute_tangent_field",permalink:"/fmmax/API/vector_fourier.compute_tangent_field"}},l={},u=[{value:"<code>vector_fourier.compute_gradient</code>",id:"vector_fouriercompute_gradient",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],p={toc:u},d="wrapper";function f(e){let{components:t,...r}=e;return(0,o.kt)(d,(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h3",{id:"vector_fouriercompute_gradient"},(0,o.kt)("inlineCode",{parentName:"h3"},"vector_fourier.compute_gradient")),(0,o.kt)("p",null,"Computes the gradient of ",(0,o.kt)("inlineCode",{parentName:"p"},"arr"),"."),(0,o.kt)("p",null,"The gradient is scaled for the size of the unit cell, i.e. two unit cells with\nidentical ",(0,o.kt)("inlineCode",{parentName:"p"},"arr")," and scaled ",(0,o.kt)("inlineCode",{parentName:"p"},"primitive_lattice_vectors")," will have identical\ngradient."),(0,o.kt)("h4",{id:"args"},"Args:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"arr"),": The array for which the gradient is sought."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"primitive_lattice_vectors"),": Defines the unit cell coordinates.")),(0,o.kt)("h4",{id:"returns"},"Returns:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"None"),": The gradient, with shape ",(0,o.kt)("inlineCode",{parentName:"li"},"arr.shape + (2,)"),".")))}f.isMDXComponent=!0}}]);