class Nodo:
    def __init__(self, nombre):
        self.nombre = nombre
        self.izquierda = None
        self.derecha = None

class Arbol:
    def __init__(self):
        self.raiz = None

    def vacio(self):
        return self.raiz is None

    def buscarNodo(self, nombre):
        return self._buscarNodoRec(self.raiz, nombre)

    def _buscarNodoRec(self, nodo, nombre):
        if nodo is None or nodo.nombre == nombre:
            return nodo
        if nombre < nodo.nombre:
            return self._buscarNodoRec(nodo.izquierda, nombre)
        else:
            return self._buscarNodoRec(nodo.derecha, nombre)

    def insertar(self, nombre):
        if self.vacio():
            self.raiz = Nodo(nombre)
        else:
            self._insertarRec(self.raiz, nombre)

    def _insertarRec(self, nodo, nombre):
        if nombre < nodo.nombre:
            if nodo.izquierda is None:
                nodo.izquierda = Nodo(nombre)
            else:
                self._insertarRec(nodo.izquierda, nombre)
        elif nombre > nodo.nombre:
            if nodo.derecha is None:
                nodo.derecha = Nodo(nombre)
            else:
                self._insertarRec(nodo.derecha, nombre)

arbol = Arbol()
arbol.insertar("Carlos")
arbol.insertar("Ana")
arbol.insertar("Pedro")
arbol.insertar("Maria")

nodo = arbol.buscarNodo("Ana")
if nodo:
    print(f"Nodo encontrado: {nodo.nombre}")
else:
    print("Nodo no encontrado")

print(f"¿El árbol está vacío? {arbol.vacio()}")