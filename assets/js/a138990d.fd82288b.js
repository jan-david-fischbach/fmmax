"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6481],{3905:(e,t,r)=>{r.d(t,{Zo:()=>f,kt:()=>d});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var c=n.createContext({}),u=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},f=function(e){var t=u(e.components);return n.createElement(c.Provider,{value:t},e.children)},p="mdxType",s={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,f=l(e,["components","mdxType","originalType","parentName"]),p=u(r),m=o,d=p["".concat(c,".").concat(m)]||p[m]||s[m]||i;return r?n.createElement(d,a(a({ref:t},f),{},{components:r})):n.createElement(d,a({ref:t},f))}));function d(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=r.length,a=new Array(i);a[0]=m;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l[p]="string"==typeof e?e:o,a[1]=l;for(var u=2;u<i;u++)a[u]=r[u];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},3330:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>c,contentTitle:()=>a,default:()=>s,frontMatter:()=>i,metadata:()=>l,toc:()=>u});var n=r(7462),o=(r(7294),r(3905));const i={id:"vector_fourier.compute_tangent_field"},a=void 0,l={unversionedId:"API/vector_fourier.compute_tangent_field",id:"API/vector_fourier.compute_tangent_field",title:"vector_fourier.compute_tangent_field",description:"vectorfourier.computetangent_field",source:"@site/docs/API/vector_fourier.compute_tangent_field.md",sourceDirName:"API",slug:"/API/vector_fourier.compute_tangent_field",permalink:"/fmmax/API/vector_fourier.compute_tangent_field",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/vector_fourier.compute_tangent_field.md",tags:[],version:"current",frontMatter:{id:"vector_fourier.compute_tangent_field"},sidebar:"APISidebar",previous:{title:"vector_fourier.compute_gradient",permalink:"/fmmax/API/vector_fourier.compute_gradient"},next:{title:"vector_fourier.field_loss_value_jac_and_hessian",permalink:"/fmmax/API/vector_fourier.field_loss_value_jac_and_hessian"}},c={},u=[{value:"<code>vector_fourier.compute_tangent_field</code>",id:"vector_fouriercompute_tangent_field",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],f={toc:u},p="wrapper";function s(e){let{components:t,...r}=e;return(0,o.kt)(p,(0,n.Z)({},f,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h3",{id:"vector_fouriercompute_tangent_field"},(0,o.kt)("inlineCode",{parentName:"h3"},"vector_fourier.compute_tangent_field")),(0,o.kt)("p",null,"Compute the tangent vector field for ",(0,o.kt)("inlineCode",{parentName:"p"},"arr"),"."),(0,o.kt)("p",null,"The calculation finds the minimum of a quadratic loss function using a single\nNewton iteration. Rather than optimizing the real-space tangent field, the\nFourier coefficients are directly optimized."),(0,o.kt)("h4",{id:"args"},"Args:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"arr"),": The array for which the normal vector field is sought."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"expansion"),": The Fourier expansion for which the field is to be optimized."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"primitive_lattice_vectors"),": Define the unit cell coordinates."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"use_jones_direct"),": Specifies whether the complex Jones field is to be sought."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"fourier_loss_weight"),": Determines the weight of the loss term penalizing\nFourier terms corresponding to high frequencies."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"steps"),": The number of Newton iterations to carry out. Generally, the default\nsingle iteration is sufficient to obtain converged fields.")),(0,o.kt)("h4",{id:"returns"},"Returns:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"None"),": The normal field, ",(0,o.kt)("inlineCode",{parentName:"li"},"(tx, ty)"),".")))}s.isMDXComponent=!0}}]);