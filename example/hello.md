


This is the `hello.md` file. It's pretty useful and contains all of the basic stuff. I have just updated the hello.md file.

# Header One
## Header Two
### Header Three
#### Header Four


In this new paragraph I'm gonna do some *italics* for emphasis followed by some **bolded terms** and maybe even some new stuff ~~or perhaps not, it's a bad idea~~. Regardless, it can all be simplified by $f(x) = \sin(x) + 2 \cdot \Theta \pi$. This equation can also be implemented in some typescript code:


```ts
console.log("Hello world!");


function sum(n: number): number {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += n;
    }
    return sum
}
```


We can also do some math blocks. The fundamental theorem of calculus is:

$$
\int_{a}^{b}f(x) dx = F(b) - F(a)
$$



Blah blah blah. This is a markdown file that is working correctly. It is being rendered.


The formula for the dot product of a vector in $\mathbb{R}^n$:

$$
\vec{u} \cdot \vec{v} = \sum_{i = 1}^{n} u_{i} v_{i}
$$


And the cross product (which only exists in $\mathbb{R}^3$) is:

$$
\vec{u} \times \vec{v} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix}
$$

Or can be expanded out to

$$
\vec{u} \times \vec{v} = (u_2v_3 - u_3v_2)\vec{i} - (u_1v_3 - u_3v_1)\vec{j} + (u_1v_2 - u_2v_1)\vec{k}
$$

or written as:

$$
\vec{u} \times \vec{v} = \begin{pmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{pmatrix}
$$
